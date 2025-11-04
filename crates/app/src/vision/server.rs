//! Actix Web preview server exposing the HUD, MJPEG stream, and detection APIs.
//!
//! The server runs on a dedicated thread to keep the pipeline hot path free from
//! Actix runtime concerns. It surfaces live frames, a small history buffer, and
//! SSE streams for downstream consumers.

use std::time::Duration;

use actix_web::{
    App, HttpResponse, HttpServer,
    http::header,
    web::{self, Bytes},
};
use anyhow::{Context, Result};
use async_stream::stream;
use serde::Deserialize;
use serde_json::to_string;
use tokio::sync::oneshot;
use tracing::error;

use crate::vision::data::{DetectionsResponse, FrameHistory, FramePacket, SharedFrame};

/// Shared state backing HTTP handlers.
pub(crate) struct ServerState {
    pub(crate) latest: SharedFrame,
    pub(crate) history: FrameHistory,
}

#[derive(Default)]
/// Handle for the preview server thread.
pub(crate) struct PreviewServer {
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PreviewServer {
    /// Signal the server to stop and block until the thread exits.
    pub(crate) fn stop(self) {
        if let Some(tx) = self.shutdown {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle {
            let _ = handle.join();
        }
    }
}

#[derive(Deserialize)]
struct FrameQuery {
    frame: Option<u64>,
}

/// Spawn the preview server thread and return a handle that can stop it.
pub(crate) fn spawn_preview_server(
    shared: SharedFrame,
    history: FrameHistory,
) -> Result<PreviewServer> {
    let server_shared = shared.clone();
    let server_history = history.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = std::thread::Builder::new()
        .name("vision-preview-server".into())
        .spawn(move || {
            if let Err(err) = actix_web::rt::System::new().block_on(async move {
                let server = HttpServer::new(move || {
                    App::new()
                        .app_data(web::Data::new(ServerState {
                            latest: server_shared.clone(),
                            history: server_history.clone(),
                        }))
                        .route("/atak", web::get().to(atak_route))
                        .route("/", web::get().to(index_route))
                        .route("/frame.jpg", web::get().to(frame_handler))
                        .route("/stream.mjpg", web::get().to(stream_handler))
                        .route("/detections", web::get().to(detections_handler))
                        .route(
                            "/stream_detections",
                            web::get().to(stream_detections_handler),
                        )
                })
                .bind(("0.0.0.0", 8080))?
                .run();

                let srv_handle = server.handle();
                actix_web::rt::spawn(async move {
                    let _ = shutdown_rx.await;
                    srv_handle.stop(true).await;
                });

                server.await
            }) {
                error!("HTTP server error: {err}");
            }
        })
        .context("Failed to spawn preview server thread")?;
    Ok(PreviewServer {
        shutdown: Some(shutdown_tx),
        handle: Some(handle),
    })
}

/// Fetch the latest encoded frame from the shared pointer.
fn latest_frame(shared: &SharedFrame) -> Option<FramePacket> {
    match shared.lock() {
        Ok(guard) => guard.clone(),
        Err(_) => None,
    }
}

/// Retrieve a historical frame by sequence number.
fn history_frame(history: &FrameHistory, frame_number: u64) -> Option<FramePacket> {
    match history.lock() {
        Ok(buffer) => buffer
            .iter()
            .find(|packet| packet.frame_number == frame_number)
            .cloned(),
        Err(_) => None,
    }
}

/// Return a single JPEG frame by sequence number or the latest frame.
async fn frame_handler(
    query: web::Query<FrameQuery>,
    state: web::Data<ServerState>,
) -> HttpResponse {
    if let Some(requested) = query.frame {
        if let Some(packet) = history_frame(&state.history, requested) {
            return HttpResponse::Ok()
                .content_type("image/jpeg")
                .body(packet.jpeg);
        } else if let Some(latest) = latest_frame(&state.latest) {
            return HttpResponse::Ok()
                .append_header((
                    header::WARNING,
                    format!(
                        "299 vision \"frame {} not buffered; returning latest {}\"",
                        requested, latest.frame_number
                    ),
                ))
                .content_type("image/jpeg")
                .body(latest.jpeg);
        } else {
            return HttpResponse::NoContent().finish();
        }
    }

    match latest_frame(&state.latest) {
        Some(packet) => HttpResponse::Ok()
            .content_type("image/jpeg")
            .body(packet.jpeg),
        None => HttpResponse::NoContent().finish(),
    }
}

/// Stream the MJPEG feed over a multipart response.
async fn stream_handler(state: web::Data<ServerState>) -> HttpResponse {
    let state = state.clone();
    let stream = stream! {
        let mut interval = actix_web::rt::time::interval(Duration::from_millis(33));
        loop {
            interval.tick().await;
            let frame = state
                .latest
                .lock()
                .ok()
                .and_then(|guard| guard.clone());
            if let Some(packet) = frame {
                let mut payload = Vec::with_capacity(packet.jpeg.len() + 64);
                payload.extend_from_slice(b"--frame\r\n");
                payload.extend_from_slice(
                    format!("X-Sequence: {}\r\n", packet.frame_number).as_bytes(),
                );
                payload.extend_from_slice(b"Content-Type: image/jpeg\r\n\r\n");
                payload.extend_from_slice(&packet.jpeg);
                payload.extend_from_slice(b"\r\n");
                yield Ok::<Bytes, actix_web::Error>(Bytes::from(payload));
            }
        }
    };

    HttpResponse::Ok()
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_HEADERS, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_METHODS, "GET"))
        .insert_header((header::ACCESS_CONTROL_EXPOSE_HEADERS, "Content-Type"))
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
        .streaming(stream)
}

/// Serve the default HUD HTML.
async fn index_route() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(crate::html::hud_html::HUD_INDEX_HTML)
}

/// Serve the ATAK-style HUD.
async fn atak_route() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(crate::html::atak::HUD_ATAK_HTML)
}

/// Return the most recent detection snapshot as JSON.
async fn detections_handler(state: web::Data<ServerState>) -> HttpResponse {
    let guard = match state.latest.lock() {
        Ok(guard) => guard,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };
    if let Some(ref packet) = *guard {
        HttpResponse::Ok().json(DetectionsResponse {
            timestamp_ms: packet.timestamp_ms,
            frame_number: packet.frame_number,
            fps: packet.fps,
            detections: &packet.detections,
        })
    } else {
        HttpResponse::NoContent().finish()
    }
}

/// Stream detection snapshots as Server-Sent Events.
async fn stream_detections_handler(state: web::Data<ServerState>) -> HttpResponse {
    let state = state.clone();
    let stream = stream! {
        yield Ok::<Bytes, actix_web::Error>(Bytes::from_static(b"retry: 500\n\n"));
        let mut interval = actix_web::rt::time::interval(Duration::from_millis(250));
        loop {
            interval.tick().await;
            let snapshot = state
                .latest
                .lock()
                .ok()
                .and_then(|guard| guard.clone());
            if let Some(packet) = snapshot {
                let payload = DetectionsResponse {
                    timestamp_ms: packet.timestamp_ms,
                    frame_number: packet.frame_number,
                    fps: packet.fps,
                    detections: &packet.detections,
                };
                match to_string(&payload) {
                    Ok(json) => {
                        let mut sse_chunk = String::with_capacity(json.len() + 32);
                        sse_chunk.push_str("id: ");
                        sse_chunk.push_str(&packet.frame_number.to_string());
                        sse_chunk.push('\n');
                        sse_chunk.push_str("data: ");
                        sse_chunk.push_str(&json);
                        sse_chunk.push_str("\n\n");
                        yield Ok::<Bytes, actix_web::Error>(Bytes::from(sse_chunk));
                    }
                    Err(err) => {
                        let error_chunk = format!("event: error\ndata: {}\n\n", err);
                        yield Ok::<Bytes, actix_web::Error>(Bytes::from(error_chunk));
                    }
                }
            } else {
                yield Ok::<Bytes, actix_web::Error>(Bytes::from_static(b": keep-alive\n\n"));
            }
        }
    };

    HttpResponse::Ok()
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_HEADERS, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_METHODS, "GET"))
        .insert_header((header::ACCESS_CONTROL_EXPOSE_HEADERS, "Content-Type"))
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Content-Type", "text/event-stream"))
        .append_header(("Connection", "keep-alive"))
        .streaming(stream)
}

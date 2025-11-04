//! Debug gizmos visualising the camera rig and projection geometry.

use bevy::prelude::*;

use crate::scene::{
    CAMERA_HEAD_HEIGHT, CameraFeedPlane, PLANE_ASPECT_RATIO, PLANE_WIDTH, RIG_ROOT,
};

const ROOT_AXIS_LEN: f32 = 0.35;
const FRUSTUM_COLOR: Color = Color::srgb(0.2, 0.8, 1.0);
const PLANE_COLOR: Color = Color::srgb(0.95, 0.65, 0.1);
const CAMERA_LINK_COLOR: Color = Color::WHITE;

/// Adds systems responsible for drawing debug gizmos around the camera rig.
pub struct CameraRigGizmosPlugin;

impl Plugin for CameraRigGizmosPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, draw_camera_rig_gizmos);
    }
}

/// Draw axes, frustum lines, and screen plane alignment.
fn draw_camera_rig_gizmos(
    mut gizmos: Gizmos,
    plane_query: Query<&GlobalTransform, With<CameraFeedPlane>>,
) {
    let Some(plane_transform) = plane_query.iter().next() else {
        return;
    };

    let rig_tip = RIG_ROOT + Vec3::Y * CAMERA_HEAD_HEIGHT;
    gizmos.sphere(RIG_ROOT, 0.04, Color::srgb(1.0, 0.2, 0.2));
    gizmos.line(RIG_ROOT, rig_tip, CAMERA_LINK_COLOR);
    draw_root_axes(&mut gizmos, RIG_ROOT);

    let plane_corners = plane_corners_world(plane_transform);
    draw_loop(&mut gizmos, &plane_corners, PLANE_COLOR);
    let plane_center = plane_corners
        .iter()
        .copied()
        .fold(Vec3::ZERO, |acc, point| acc + point)
        * 0.25;
    gizmos.line(rig_tip, plane_center, CAMERA_LINK_COLOR);
    for &corner in &plane_corners {
        gizmos.line(rig_tip, corner, FRUSTUM_COLOR);
    }
}

/// Draw a closed loop through the supplied corners.
fn draw_loop(gizmos: &mut Gizmos, corners: &[Vec3; 4], color: Color) {
    let mut points = Vec::with_capacity(5);
    points.extend_from_slice(corners);
    points.push(corners[0]);
    gizmos.linestrip(points.iter().copied(), color);
}

/// Draw XYZ axes centered at the supplied origin.
fn draw_root_axes(gizmos: &mut Gizmos, origin: Vec3) {
    let axis = Vec3::X * ROOT_AXIS_LEN;
    gizmos.line(origin - axis, origin + axis, Color::srgb(1.0, 0.1, 0.1));
    let axis = Vec3::Y * ROOT_AXIS_LEN;
    gizmos.line(origin - axis, origin + axis, Color::srgb(0.1, 1.0, 0.1));
    let axis = Vec3::Z * ROOT_AXIS_LEN;
    gizmos.line(origin - axis, origin + axis, Color::srgb(0.1, 0.4, 1.0));
}

/// Compute plane corners in world space given the plane transform.
fn plane_corners_world(plane_transform: &GlobalTransform) -> [Vec3; 4] {
    let half_width = PLANE_WIDTH * 0.5;
    let half_height = (PLANE_WIDTH / PLANE_ASPECT_RATIO) * 0.5;
    let local_corners = [
        Vec3::new(-half_width, 0.0, -half_height),
        Vec3::new(half_width, 0.0, -half_height),
        Vec3::new(half_width, 0.0, half_height),
        Vec3::new(-half_width, 0.0, half_height),
    ];
    let matrix = Mat4::from(plane_transform.affine());
    local_corners.map(|corner| matrix.transform_point3(corner))
}

//! Measure received frames over elapsed time, including network pauses.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// A two-second moving window. Keep the arrival immediately before the window
/// so every counted interval has both endpoints, including across long gaps.
#[derive(Default)]
pub(crate) struct FrameRate {
    arrivals: VecDeque<Instant>,
}

impl FrameRate {
    pub(crate) fn record(&mut self, now: Instant) -> f32 {
        self.arrivals.push_back(now);
        while self.arrivals.len() > 2
            && now.duration_since(self.arrivals[1]) >= Duration::from_secs(2)
        {
            self.arrivals.pop_front();
        }
        let elapsed = now.duration_since(self.arrivals[0]).as_secs_f32();
        if elapsed > 0.0 {
            (self.arrivals.len() - 1) as f32 / elapsed
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_delivery_does_not_inflate_frame_rate() {
        let start = Instant::now();
        let mut meter = FrameRate::default();
        meter.record(start);
        // Thirty frames arrive per second, in groups of ten 1 ms apart.
        let mut fps = 0.0;
        for burst in 1..=15 {
            for frame in 0..10 {
                fps = meter.record(start + Duration::from_micros(burst * 333_333 + frame * 1_000));
            }
        }
        assert!((fps - 30.0).abs() < 1.0, "reported {fps} fps");
    }

    #[test]
    fn steady_stream_and_long_pause_are_measured() {
        let start = Instant::now();
        let mut meter = FrameRate::default();
        assert_eq!(meter.record(start), 0.0);
        let mut fps = 0.0;
        for frame in 1..=100 {
            fps = meter.record(start + Duration::from_millis(frame * 50));
        }
        assert!((fps - 20.0).abs() < 0.01);
        assert!((meter.record(start + Duration::from_secs(8)) - 1.0 / 3.0).abs() < 0.01);
    }
}

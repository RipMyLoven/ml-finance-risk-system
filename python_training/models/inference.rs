//! ONNX Model Inference for Rust
//! 
//! Generated: 2026-01-26 23:33
//! Features: 342
//!
//! Cargo.toml:
//! ```toml
//! [dependencies]
//! ort = "2.0"
//! ndarray = "0.16"
//! ```

use ort::{GraphOptimizationLevel, Session};
use ndarray::Array2;
use std::error::Error;

pub const NUM_FEATURES: usize = 342;

/// Load ONNX model
pub fn load_model(path: &str) -> Result<Session, Box<dyn Error>> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?;
    Ok(session)
}

/// Run inference
/// Returns probability of price going UP
pub fn predict(session: &Session, features: &[f32]) -> Result<f32, Box<dyn Error>> {
    if features.len() != NUM_FEATURES {
        return Err(format!(
            "Expected {} features, got {}", 
            NUM_FEATURES, 
            features.len()
        ).into());
    }
    
    let input = Array2::from_shape_vec((1, NUM_FEATURES), features.to_vec())?;
    let outputs = session.run(ort::inputs!["input" => input.view()]?)?;
    
    // Output[1] contains probabilities [[prob_down, prob_up]]
    let probs = outputs[1].try_extract_tensor::<f32>()?;
    let prob_up = probs[[0, 1]];
    
    Ok(prob_up)
}

/// Trading signal based on probability
pub fn get_signal(prob: f32) -> &'static str {
    if prob > 0.65 {
        "STRONG_BUY"
    } else if prob > 0.55 {
        "BUY"
    } else if prob < 0.35 {
        "STRONG_SELL"
    } else if prob < 0.45 {
        "SELL"
    } else {
        "HOLD"
    }
}

// Feature names (in order):
/*
  0: return_1
  1: return_3
  2: return_5
  3: return_10
  4: return_20
  5: cum_return_5
  6: cum_return_10
  7: bar_position
  8: bar_range_pct
  9: bar_body_pct
  10: upper_shadow
  11: lower_shadow
  12: gap
  13: volume_ratio
  14: volume_trend
  15: volume_change
  16: volume_accel
  17: trades_ratio
  18: buy_pressure
  19: buy_pressure_sma
  20: buy_pressure_change
  21: pv_corr
  22: price_vs_sma5
  23: price_vs_sma10
  24: price_vs_sma20
  25: price_vs_sma50
  26: sma10_slope
  27: sma20_slope
  28: ema_cross
  29: ema_cross_slope
  ... and 312 more
*/

//! Rolling window buffer for temporal coherence tracking.
//!
//! Provides a memory-efficient circular buffer implementation for maintaining
//! a sliding window of recent values, useful for computing temporal coherence
//! and variance metrics.
//!
//! # Example
//!
//! ```ignore
//! use context_graph_utl::coherence::{RollingWindow, WindowConfig};
//!
//! let mut window: RollingWindow<f32> = RollingWindow::new(5);
//!
//! window.push(1.0);
//! window.push(2.0);
//! window.push(3.0);
//!
//! assert_eq!(window.len(), 3);
//! assert_eq!(window.average(), Some(2.0));
//! ```

use std::collections::VecDeque;

/// Configuration for rolling window behavior.
///
/// Controls window size and minimum sample requirements for
/// various statistical computations.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Maximum number of items to keep in the window.
    pub size: usize,

    /// Minimum number of samples required for average computation.
    pub min_samples_for_average: usize,

    /// Minimum number of samples required for variance computation.
    pub min_samples_for_variance: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            size: 100,
            min_samples_for_average: 1,
            min_samples_for_variance: 2,
        }
    }
}

impl WindowConfig {
    /// Create a new window configuration with the specified size.
    pub fn with_size(size: usize) -> Self {
        Self {
            size,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, or an error message describing the issue.
    pub fn validate(&self) -> Result<(), String> {
        if self.size == 0 {
            return Err("Window size must be > 0".to_string());
        }
        if self.min_samples_for_average == 0 {
            return Err("min_samples_for_average must be > 0".to_string());
        }
        if self.min_samples_for_variance < 2 {
            return Err("min_samples_for_variance must be >= 2".to_string());
        }
        Ok(())
    }
}

/// A generic rolling window buffer with configurable size.
///
/// Implements a circular buffer that automatically drops the oldest
/// item when the window is full and a new item is added.
///
/// # Type Parameters
///
/// - `T`: The type of items stored in the window.
#[derive(Debug, Clone)]
pub struct RollingWindow<T> {
    /// Internal buffer using VecDeque for efficient front/back operations.
    buffer: VecDeque<T>,

    /// Maximum capacity of the window.
    capacity: usize,
}

impl<T> RollingWindow<T> {
    /// Create a new rolling window with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of items the window can hold.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::RollingWindow;
    ///
    /// let window: RollingWindow<f32> = RollingWindow::new(10);
    /// assert_eq!(window.capacity(), 10);
    /// ```
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Window capacity must be > 0");
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Create a new rolling window from a [`WindowConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Window configuration specifying size and thresholds.
    pub fn from_config(config: &WindowConfig) -> Self {
        Self::new(config.size)
    }

    /// Add an item to the window.
    ///
    /// If the window is at capacity, the oldest item is removed first.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to add to the window.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::RollingWindow;
    ///
    /// let mut window: RollingWindow<i32> = RollingWindow::new(3);
    /// window.push(1);
    /// window.push(2);
    /// window.push(3);
    /// window.push(4); // Removes 1, adds 4
    ///
    /// assert_eq!(window.len(), 3);
    /// ```
    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    /// Get the current number of items in the window.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if the window is at full capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Get the maximum capacity of the window.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all items from the window.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get an iterator over the items in the window.
    ///
    /// Items are iterated from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    /// Get the most recently added item.
    pub fn last(&self) -> Option<&T> {
        self.buffer.back()
    }

    /// Get the oldest item in the window.
    pub fn first(&self) -> Option<&T> {
        self.buffer.front()
    }

    /// Get an item by index (0 = oldest).
    pub fn get(&self, index: usize) -> Option<&T> {
        self.buffer.get(index)
    }
}

impl<T: Clone> RollingWindow<T> {
    /// Convert the window contents to a vector.
    ///
    /// Items are ordered from oldest to newest.
    pub fn to_vec(&self) -> Vec<T> {
        self.buffer.iter().cloned().collect()
    }
}

impl RollingWindow<f32> {
    /// Compute the average of all values in the window.
    ///
    /// # Returns
    ///
    /// `Some(average)` if the window contains at least one value,
    /// `None` if the window is empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::RollingWindow;
    ///
    /// let mut window: RollingWindow<f32> = RollingWindow::new(10);
    /// window.push(1.0);
    /// window.push(2.0);
    /// window.push(3.0);
    ///
    /// assert_eq!(window.average(), Some(2.0));
    /// ```
    pub fn average(&self) -> Option<f32> {
        if self.buffer.is_empty() {
            return None;
        }
        let sum: f32 = self.buffer.iter().sum();
        Some(sum / self.buffer.len() as f32)
    }

    /// Compute the variance of all values in the window.
    ///
    /// Uses the population variance formula (divides by n, not n-1).
    ///
    /// # Returns
    ///
    /// `Some(variance)` if the window contains at least 2 values,
    /// `None` if the window has fewer than 2 values.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::RollingWindow;
    ///
    /// let mut window: RollingWindow<f32> = RollingWindow::new(10);
    /// window.push(1.0);
    /// window.push(2.0);
    /// window.push(3.0);
    ///
    /// let variance = window.variance().unwrap();
    /// assert!((variance - 0.6667).abs() < 0.01);
    /// ```
    pub fn variance(&self) -> Option<f32> {
        if self.buffer.len() < 2 {
            return None;
        }

        let mean = self.average()?;
        let sum_squared_diff: f32 = self
            .buffer
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();

        Some(sum_squared_diff / self.buffer.len() as f32)
    }

    /// Compute the standard deviation of all values in the window.
    ///
    /// # Returns
    ///
    /// `Some(std_dev)` if variance can be computed,
    /// `None` otherwise.
    pub fn std_dev(&self) -> Option<f32> {
        self.variance().map(|v| v.sqrt())
    }

    /// Compute the minimum value in the window.
    pub fn min(&self) -> Option<f32> {
        self.buffer
            .iter()
            .copied()
            .reduce(|a, b| if a < b { a } else { b })
    }

    /// Compute the maximum value in the window.
    pub fn max(&self) -> Option<f32> {
        self.buffer
            .iter()
            .copied()
            .reduce(|a, b| if a > b { a } else { b })
    }

    /// Compute the range (max - min) of values in the window.
    pub fn range(&self) -> Option<f32> {
        match (self.min(), self.max()) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        }
    }
}

impl RollingWindow<Vec<f32>> {
    /// Compute the element-wise average of all vectors in the window.
    ///
    /// All vectors must have the same dimension.
    ///
    /// # Returns
    ///
    /// `Some(average_vec)` if the window is non-empty and all vectors
    /// have consistent dimensions, `None` otherwise.
    pub fn average_vec(&self) -> Option<Vec<f32>> {
        if self.buffer.is_empty() {
            return None;
        }

        let dim = self.buffer.front()?.len();
        if dim == 0 {
            return None;
        }

        // Verify all vectors have the same dimension
        if !self.buffer.iter().all(|v| v.len() == dim) {
            return None;
        }

        let mut result = vec![0.0f32; dim];
        let count = self.buffer.len() as f32;

        for vec in self.buffer.iter() {
            for (i, &val) in vec.iter().enumerate() {
                result[i] += val;
            }
        }

        for val in &mut result {
            *val /= count;
        }

        Some(result)
    }

    /// Compute element-wise variance across all vectors in the window.
    ///
    /// # Returns
    ///
    /// `Some(variance_vec)` where each element is the variance of that
    /// dimension across all vectors, `None` if fewer than 2 vectors.
    pub fn variance_vec(&self) -> Option<Vec<f32>> {
        if self.buffer.len() < 2 {
            return None;
        }

        let mean = self.average_vec()?;
        let dim = mean.len();
        let count = self.buffer.len() as f32;

        let mut variance = vec![0.0f32; dim];

        for vec in self.buffer.iter() {
            for (i, &val) in vec.iter().enumerate() {
                let diff = val - mean[i];
                variance[i] += diff * diff;
            }
        }

        for val in &mut variance {
            *val /= count;
        }

        Some(variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert_eq!(config.size, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_window_config_with_size() {
        let config = WindowConfig::with_size(50);
        assert_eq!(config.size, 50);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_window_config_validation() {
        let invalid_size = WindowConfig {
            size: 0,
            ..Default::default()
        };
        assert!(invalid_size.validate().is_err());

        let invalid_avg = WindowConfig {
            min_samples_for_average: 0,
            ..Default::default()
        };
        assert!(invalid_avg.validate().is_err());

        let invalid_var = WindowConfig {
            min_samples_for_variance: 1,
            ..Default::default()
        };
        assert!(invalid_var.validate().is_err());
    }

    #[test]
    fn test_rolling_window_new() {
        let window: RollingWindow<i32> = RollingWindow::new(5);
        assert_eq!(window.capacity(), 5);
        assert!(window.is_empty());
        assert!(!window.is_full());
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_rolling_window_zero_capacity() {
        let _: RollingWindow<i32> = RollingWindow::new(0);
    }

    #[test]
    fn test_rolling_window_push() {
        let mut window: RollingWindow<i32> = RollingWindow::new(3);

        window.push(1);
        assert_eq!(window.len(), 1);
        assert_eq!(window.first(), Some(&1));
        assert_eq!(window.last(), Some(&1));

        window.push(2);
        window.push(3);
        assert_eq!(window.len(), 3);
        assert!(window.is_full());

        // Should evict oldest item
        window.push(4);
        assert_eq!(window.len(), 3);
        assert_eq!(window.first(), Some(&2));
        assert_eq!(window.last(), Some(&4));
    }

    #[test]
    fn test_rolling_window_clear() {
        let mut window: RollingWindow<i32> = RollingWindow::new(5);
        window.push(1);
        window.push(2);
        window.clear();
        assert!(window.is_empty());
    }

    #[test]
    fn test_rolling_window_get() {
        let mut window: RollingWindow<i32> = RollingWindow::new(5);
        window.push(10);
        window.push(20);
        window.push(30);

        assert_eq!(window.get(0), Some(&10));
        assert_eq!(window.get(1), Some(&20));
        assert_eq!(window.get(2), Some(&30));
        assert_eq!(window.get(3), None);
    }

    #[test]
    fn test_rolling_window_iter() {
        let mut window: RollingWindow<i32> = RollingWindow::new(5);
        window.push(1);
        window.push(2);
        window.push(3);

        let items: Vec<i32> = window.iter().copied().collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn test_rolling_window_to_vec() {
        let mut window: RollingWindow<i32> = RollingWindow::new(5);
        window.push(1);
        window.push(2);
        window.push(3);

        assert_eq!(window.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_f32_average() {
        let mut window: RollingWindow<f32> = RollingWindow::new(10);
        assert_eq!(window.average(), None);

        window.push(1.0);
        assert_eq!(window.average(), Some(1.0));

        window.push(2.0);
        window.push(3.0);
        assert_eq!(window.average(), Some(2.0));
    }

    #[test]
    fn test_f32_variance() {
        let mut window: RollingWindow<f32> = RollingWindow::new(10);
        assert_eq!(window.variance(), None);

        window.push(1.0);
        assert_eq!(window.variance(), None);

        window.push(2.0);
        window.push(3.0);

        // Variance of [1, 2, 3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3 ≈ 0.667
        let variance = window.variance().unwrap();
        assert!((variance - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_f32_std_dev() {
        let mut window: RollingWindow<f32> = RollingWindow::new(10);
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);

        let std_dev = window.std_dev().unwrap();
        let variance = window.variance().unwrap();
        assert!((std_dev - variance.sqrt()).abs() < 0.0001);
    }

    #[test]
    fn test_f32_min_max_range() {
        let mut window: RollingWindow<f32> = RollingWindow::new(10);
        window.push(3.0);
        window.push(1.0);
        window.push(5.0);
        window.push(2.0);

        assert_eq!(window.min(), Some(1.0));
        assert_eq!(window.max(), Some(5.0));
        assert_eq!(window.range(), Some(4.0));
    }

    #[test]
    fn test_vec_average() {
        let mut window: RollingWindow<Vec<f32>> = RollingWindow::new(10);
        assert_eq!(window.average_vec(), None);

        window.push(vec![1.0, 2.0, 3.0]);
        window.push(vec![3.0, 4.0, 5.0]);

        let avg = window.average_vec().unwrap();
        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec_variance() {
        let mut window: RollingWindow<Vec<f32>> = RollingWindow::new(10);
        assert_eq!(window.variance_vec(), None);

        window.push(vec![1.0, 2.0]);
        assert_eq!(window.variance_vec(), None);

        window.push(vec![3.0, 4.0]);

        let variance = window.variance_vec().unwrap();
        // For dim 0: values [1, 3], mean=2, variance=(1+1)/2=1
        // For dim 1: values [2, 4], mean=3, variance=(1+1)/2=1
        assert_eq!(variance, vec![1.0, 1.0]);
    }

    #[test]
    fn test_vec_dimension_mismatch() {
        let mut window: RollingWindow<Vec<f32>> = RollingWindow::new(10);
        window.push(vec![1.0, 2.0]);
        window.push(vec![3.0, 4.0, 5.0]); // Different dimension

        // Should return None due to dimension mismatch
        assert_eq!(window.average_vec(), None);
    }

    #[test]
    fn test_from_config() {
        let config = WindowConfig::with_size(25);
        let window: RollingWindow<f32> = RollingWindow::from_config(&config);
        assert_eq!(window.capacity(), 25);
    }
}

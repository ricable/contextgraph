//! Validation logic for SemanticFingerprint.

use super::constants::{
    E10_DIM, E11_DIM, E12_TOKEN_DIM, E13_SPLADE_VOCAB, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E6_SPARSE_VOCAB, E7_DIM, E8_DIM, E9_DIM,
};
use super::fingerprint::SemanticFingerprint;

impl SemanticFingerprint {
    /// Validate all embeddings have correct dimensions.
    ///
    /// This method performs a fail-fast validation of all embedding dimensions.
    /// Returns an error immediately if any dimension is incorrect.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All embeddings have correct dimensions
    /// * `Err(String)` - Description of validation failure
    ///
    /// # Validation Rules
    ///
    /// 1. E1: Must have exactly 1024 dimensions
    /// 2. E2-E4: Must have exactly 512 dimensions each
    /// 3. E5: Must have exactly 768 dimensions
    /// 4. E6: Sparse indices must be < E6_SPARSE_VOCAB
    /// 5. E7: Must have exactly 256 dimensions
    /// 6. E8, E11: Must have exactly 384 dimensions each
    /// 7. E9: Must have exactly 10000 dimensions
    /// 8. E10: Must have exactly 768 dimensions
    /// 9. E12: Each token must have exactly 128 dimensions
    /// 10. E13: Sparse indices must be < E13_SPLADE_VOCAB
    pub fn validate(&self) -> Result<(), String> {
        self.validate_e1()?;
        self.validate_e2()?;
        self.validate_e3()?;
        self.validate_e4()?;
        self.validate_e5()?;
        self.validate_e6()?;
        self.validate_e7()?;
        self.validate_e8()?;
        self.validate_e9()?;
        self.validate_e10()?;
        self.validate_e11()?;
        self.validate_e12()?;
        self.validate_e13()?;
        Ok(())
    }

    fn validate_e1(&self) -> Result<(), String> {
        if self.e1_semantic.len() != E1_DIM {
            return Err(format!(
                "E1 semantic dimension mismatch: expected {}, got {}",
                E1_DIM,
                self.e1_semantic.len()
            ));
        }
        Ok(())
    }

    fn validate_e2(&self) -> Result<(), String> {
        if self.e2_temporal_recent.len() != E2_DIM {
            return Err(format!(
                "E2 temporal_recent dimension mismatch: expected {}, got {}",
                E2_DIM,
                self.e2_temporal_recent.len()
            ));
        }
        Ok(())
    }

    fn validate_e3(&self) -> Result<(), String> {
        if self.e3_temporal_periodic.len() != E3_DIM {
            return Err(format!(
                "E3 temporal_periodic dimension mismatch: expected {}, got {}",
                E3_DIM,
                self.e3_temporal_periodic.len()
            ));
        }
        Ok(())
    }

    fn validate_e4(&self) -> Result<(), String> {
        if self.e4_temporal_positional.len() != E4_DIM {
            return Err(format!(
                "E4 temporal_positional dimension mismatch: expected {}, got {}",
                E4_DIM,
                self.e4_temporal_positional.len()
            ));
        }
        Ok(())
    }

    fn validate_e5(&self) -> Result<(), String> {
        if self.e5_causal.len() != E5_DIM {
            return Err(format!(
                "E5 causal dimension mismatch: expected {}, got {}",
                E5_DIM,
                self.e5_causal.len()
            ));
        }
        Ok(())
    }

    fn validate_e6(&self) -> Result<(), String> {
        for &idx in &self.e6_sparse.indices {
            if idx as usize >= E6_SPARSE_VOCAB {
                return Err(format!(
                    "E6 sparse index {} exceeds vocabulary size {}",
                    idx, E6_SPARSE_VOCAB
                ));
            }
        }
        if self.e6_sparse.indices.len() != self.e6_sparse.values.len() {
            return Err(format!(
                "E6 sparse indices ({}) and values ({}) length mismatch",
                self.e6_sparse.indices.len(),
                self.e6_sparse.values.len()
            ));
        }
        Ok(())
    }

    fn validate_e7(&self) -> Result<(), String> {
        if self.e7_code.len() != E7_DIM {
            return Err(format!(
                "E7 code dimension mismatch: expected {}, got {}",
                E7_DIM,
                self.e7_code.len()
            ));
        }
        Ok(())
    }

    fn validate_e8(&self) -> Result<(), String> {
        if self.e8_graph.len() != E8_DIM {
            return Err(format!(
                "E8 graph dimension mismatch: expected {}, got {}",
                E8_DIM,
                self.e8_graph.len()
            ));
        }
        Ok(())
    }

    fn validate_e9(&self) -> Result<(), String> {
        if self.e9_hdc.len() != E9_DIM {
            return Err(format!(
                "E9 hdc dimension mismatch: expected {}, got {}",
                E9_DIM,
                self.e9_hdc.len()
            ));
        }
        Ok(())
    }

    fn validate_e10(&self) -> Result<(), String> {
        if self.e10_multimodal.len() != E10_DIM {
            return Err(format!(
                "E10 multimodal dimension mismatch: expected {}, got {}",
                E10_DIM,
                self.e10_multimodal.len()
            ));
        }
        Ok(())
    }

    fn validate_e11(&self) -> Result<(), String> {
        if self.e11_entity.len() != E11_DIM {
            return Err(format!(
                "E11 entity dimension mismatch: expected {}, got {}",
                E11_DIM,
                self.e11_entity.len()
            ));
        }
        Ok(())
    }

    fn validate_e12(&self) -> Result<(), String> {
        for (i, token) in self.e12_late_interaction.iter().enumerate() {
            if token.len() != E12_TOKEN_DIM {
                return Err(format!(
                    "E12 late_interaction token {} dimension mismatch: expected {}, got {}",
                    i,
                    E12_TOKEN_DIM,
                    token.len()
                ));
            }
        }
        Ok(())
    }

    fn validate_e13(&self) -> Result<(), String> {
        for &idx in &self.e13_splade.indices {
            if idx as usize >= E13_SPLADE_VOCAB {
                return Err(format!(
                    "E13 splade index {} exceeds vocabulary size {}",
                    idx, E13_SPLADE_VOCAB
                ));
            }
        }
        if self.e13_splade.indices.len() != self.e13_splade.values.len() {
            return Err(format!(
                "E13 splade indices ({}) and values ({}) length mismatch",
                self.e13_splade.indices.len(),
                self.e13_splade.values.len()
            ));
        }
        Ok(())
    }
}

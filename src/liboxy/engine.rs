//! This module contains the ONNX Execution engine and its internals.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputs, SessionOutputs};

use std::sync::Arc;

use anyhow::Result;

use crate::vision::VisionModel;

type PreprocessFn<I> =
    Box<dyn for<'a> Fn(&'a I) -> Result<SessionInputs<'static, 'static, 1>> + Send + Sync>;
type PostprocessFn<O> = Box<dyn for<'s> Fn(SessionOutputs<'s>) -> Result<O> + Send + Sync>;

/// Main Onnx Inference struct in `oxypipe` provides a managed onnx session via the `ort` crate.
pub struct OnnxEngine<I, O> {
    session: Session,
    preprocess: PreprocessFn<I>,
    postprocess: PostprocessFn<O>,
}

impl<I, O> OnnxEngine<I, O> {
    /// Runs the complete inference pipeline.
    pub fn run(&mut self, input: &I) -> Result<O> {
        let session_inputs = (self.preprocess)(input)?;
        let session_outputs = self.session.run(session_inputs)?;
        (self.postprocess)(session_outputs)
    }
}
/// Sane way to make a Onnx Engine
pub struct EngineBuilder<I, O> {
    model_path: String,
    preprocess: Option<PreprocessFn<I>>,
    postprocess: Option<PostprocessFn<O>>,
}

impl<I, O> EngineBuilder<I, O> {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            preprocess: None,
            postprocess: None,
        }
    }

    /// Adapts any `VisionModel` into the closure-based engine.
    pub fn with_vision_model<M>(self, model: M) -> EngineBuilder<M::Input, M::Output>
    where
        M: VisionModel,
    {
        let model = Arc::new(model);
        let m_pre = Arc::clone(&model);
        let m_post = Arc::clone(&model);

        EngineBuilder {
            model_path: self.model_path,
            preprocess: Some(Box::new(move |input| m_pre.preprocess(input))),
            postprocess: Some(Box::new(move |outputs| m_post.postprocess(outputs))),
        }
    }

    pub fn preprocess<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a I) -> Result<SessionInputs<'static, 'static, 1>> + Send + Sync + 'static,
    {
        self.preprocess = Some(Box::new(f));
        self
    }

    pub fn postprocess<F>(mut self, f: F) -> Self
    where
        F: for<'s> Fn(SessionOutputs<'s>) -> Result<O> + Send + Sync + 'static,
    {
        self.postprocess = Some(Box::new(f));
        self
    }

    pub fn build(self) -> Result<OnnxEngine<I, O>> {
        // ort's SessionBuilder errors carry a non-Sync handle, so map them into
        // plain anyhow errors to keep `Result` (Send + Sync) bounds intact.
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
            .commit_from_file(self.model_path)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let preprocess = self
            .preprocess
            .ok_or_else(|| anyhow::anyhow!("Preprocess function required"))?;
        let postprocess = self
            .postprocess
            .ok_or_else(|| anyhow::anyhow!("Postprocess function required"))?;

        Ok(OnnxEngine {
            session,
            preprocess,
            postprocess,
        })
    }
}

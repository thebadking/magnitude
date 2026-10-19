//! Compare target batching on a prefix generated from a native chat fixture.
//! Both walks select their own prefix. Only the first walk's sampled next
//! token supplies the second row of the paired forward; no text is retokenized.

use crate::artifacts::Artifacts;
use crate::engine::{self, Engine, EngineOptions, Load, Sequence};
use crate::render::Rendered;
use magnitude_executor::{Demand, Operation, Outcome, TokenId, WorkKind};
use serde_json::{Value, json};
use std::path::Path;

pub struct Compared {
    pub device: String,
    pub result: Value,
}

fn write(path: &Path, result: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?)
        .map_err(|error| format!("writing {}: {error}", path.display()))
}

fn selected(sequence: &Sequence) -> Result<TokenId, String> {
    sequence.selected.ok_or_else(|| "target comparison returned no selection".into())
}

fn prefix(
    engine: &mut Engine,
    sequence: &mut Sequence,
    prompt: &[TokenId],
    anchor: usize,
) -> Result<Vec<u32>, String> {
    engine.prefill(sequence, prompt, true)?;
    let mut tokens = vec![selected(sequence)?.0];
    while sequence.position < anchor {
        let operation = Engine::forward(sequence, WorkKind::Decode,
            vec![selected(sequence)?], Demand::SELECT);
        engine.step(vec![operation], &mut [&mut *sequence], None)?;
        tokens.push(selected(sequence)?.0);
    }
    Ok(tokens)
}

pub fn run(
    artifacts: &Artifacts,
    options: &EngineOptions,
    rendered: &[Rendered],
    rendered_digest: &str,
    anchor: usize,
    model_context: usize,
    lookahead: bool,
    output: &Path,
) -> Result<Compared, String> {
    let [fixture] = rendered else {
        return Err("target-compare requires exactly one chat fixture".into());
    };
    let prompt = fixture.generation_prompt.as_deref().unwrap_or(&fixture.tokens);
    if prompt.is_empty() || anchor < prompt.len() || anchor.checked_add(2).is_none_or(|end| end > model_context) {
        return Err("target-compare needs a nonempty prompt, anchor after the prompt and context for both compared rows".into());
    }
    // Both walks use the same admitted capacity and served selection demand.
    // Full-logits export can change the readout recipe and is not enabled here.
    let mut engine = Engine::open(artifacts, options, &Load {
        drafter: false, proposals: None, context_tokens: model_context,
        decode_rows: 2, exported_logits_rows: 0, lookahead,
    })?;
    let mut result = json!({
        "status": "running", "fixture_id": fixture.fixture_id,
        "rendered_digest": rendered_digest, "template_sha256": fixture.template_sha256,
        "prompt_tokens": prompt.len(), "anchor": anchor, "model_context": model_context,
        "lookahead": lookahead, "drafting": false, "rejected_rows": 0,
        "demand": "select", "performance_claim": false, "reference_quality_claim": false,
    });
    write(output, &result)?;
    let prompt = engine::tokens(prompt);
    let mut sequential = engine.open_sequence()?;
    let control_prefix = prefix(&mut engine, &mut sequential, &prompt, anchor)?;
    result["sequential_prefix"] = json!(control_prefix);
    write(output, &result)?;
    let mut control_rows = Vec::new();
    let mut control_selections = Vec::new();
    for _ in 0..2 {
        let position = sequential.position;
        let input = selected(&sequential)?;
        let operation = Engine::forward(&sequential, WorkKind::Decode, vec![input], Demand::SELECT);
        engine.step(vec![operation], &mut [&mut sequential], None)?;
        let token = selected(&sequential)?;
        control_selections.push(token);
        control_rows.push(json!({"position": position, "input": input.0,
            "selected": token.0}));
    }
    result["sequential_rows"] = json!(control_rows);
    engine.close(sequential)?;
    write(output, &result)?;

    let mut paired = engine.open_sequence()?;
    let paired_prefix = prefix(&mut engine, &mut paired, &prompt, anchor)?;
    result["paired_prefix"] = json!(paired_prefix);
    let prefix_equal = control_prefix == paired_prefix;
    result["prefix_equal"] = json!(prefix_equal);
    if !prefix_equal {
        result["status"] = json!("failed-control");
        engine.close(paired)?;
        write(output, &result)?;
        return Err("target-compare regenerated prefixes differ; keep the failed control".into());
    }
    let first = selected(&paired)?;
    let second = control_selections[0];
    let operation = Operation::Forward {
        request: paired.request, kind: WorkKind::Verify, position: paired.position,
        conditioning: None, demand: Demand::SELECT,
        select: (0..2).map(|offset| Engine::greedy(anchor + offset)).collect(),
        committed: 2, tokens: vec![first, second], prime: None,
    };
    let (_, outcomes) = engine.step(vec![operation], &mut [&mut paired], None)?;
    let [Outcome::Forward { rows }] = outcomes.as_slice() else {
        return Err("target-compare expected one forward outcome".into());
    };
    if rows.len() != 2 { return Err("target-compare expected two paired rows".into()); }
    let paired_rows = rows.iter().enumerate().map(|(offset, row)| {
        let token = row.selected.ok_or("target-compare paired row has no selection")?.token;
        let input = [first, second][offset];
        Ok(json!({"position": anchor + offset, "input": input.0, "selected": token.0}))
    }).collect::<Result<Vec<_>, String>>()?;
    let rows_equal = control_rows == paired_rows;
    result["paired_rows"] = json!(paired_rows);
    result["rows_equal"] = json!(rows_equal);
    result["status"] = json!(if rows_equal { "passed" } else { "failed-parity" });
    engine.close(paired)?;
    write(output, &result)?;
    Ok(Compared { device: engine.device().info().name.clone(), result })
}

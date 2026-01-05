//! Tests for Modality enum.

use super::*;

#[test]
fn test_modality_default() {
    assert_eq!(Modality::default(), Modality::Text);
}

#[test]
fn test_modality_detect_rust_code() {
    assert_eq!(
        Modality::detect("fn main() { println!(\"hello\"); }"),
        Modality::Code
    );
    assert_eq!(
        Modality::detect("pub struct Foo { bar: i32 }"),
        Modality::Code
    );
    assert_eq!(Modality::detect("impl Default for Foo {}"), Modality::Code);
    assert_eq!(
        Modality::detect("use std::collections::HashMap;"),
        Modality::Code
    );
}

#[test]
fn test_modality_detect_python_code() {
    assert_eq!(Modality::detect("def hello(): pass"), Modality::Code);
    assert_eq!(Modality::detect("class Foo: pass"), Modality::Code);
    assert_eq!(Modality::detect("import os"), Modality::Code);
    assert_eq!(Modality::detect("from typing import List"), Modality::Code);
    assert_eq!(Modality::detect("async def fetch(): pass"), Modality::Code);
}

#[test]
fn test_modality_detect_javascript_code() {
    assert_eq!(Modality::detect("function foo() {}"), Modality::Code);
    assert_eq!(Modality::detect("const x = 5;"), Modality::Code);
    assert_eq!(Modality::detect("let y = 10;"), Modality::Code);
    assert_eq!(Modality::detect("var z = 'hello';"), Modality::Code);
    assert_eq!(
        Modality::detect("export default function() {}"),
        Modality::Code
    );
}

#[test]
fn test_modality_detect_structured_json() {
    assert_eq!(
        Modality::detect("{\"key\": \"value\"}"),
        Modality::Structured
    );
    assert_eq!(Modality::detect("[1, 2, 3]"), Modality::Structured);
    assert_eq!(
        Modality::detect("  { \"nested\": { } }"),
        Modality::Structured
    );
}

#[test]
fn test_modality_detect_structured_yaml() {
    assert_eq!(
        Modality::detect("key: value\nother: data"),
        Modality::Structured
    );
    assert_eq!(
        Modality::detect("name: John\nage: 30"),
        Modality::Structured
    );
}

#[test]
fn test_modality_detect_data_uri() {
    assert_eq!(
        Modality::detect("data:image/png;base64,iVBORw0KGg..."),
        Modality::Image
    );
    assert_eq!(
        Modality::detect("data:audio/mp3;base64,SUQz..."),
        Modality::Audio
    );
}

#[test]
fn test_modality_detect_plain_text() {
    assert_eq!(Modality::detect("Hello, world!"), Modality::Text);
    assert_eq!(Modality::detect("This is just a sentence."), Modality::Text);
    assert_eq!(Modality::detect("The quick brown fox"), Modality::Text);
    assert_eq!(Modality::detect(""), Modality::Text);
}

#[test]
fn test_modality_file_extensions() {
    assert!(Modality::Text.file_extensions().contains(&"txt"));
    assert!(Modality::Text.file_extensions().contains(&"md"));

    assert!(Modality::Code.file_extensions().contains(&"rs"));
    assert!(Modality::Code.file_extensions().contains(&"py"));
    assert!(Modality::Code.file_extensions().contains(&"js"));

    assert!(Modality::Image.file_extensions().contains(&"png"));
    assert!(Modality::Image.file_extensions().contains(&"jpg"));

    assert!(Modality::Audio.file_extensions().contains(&"mp3"));
    assert!(Modality::Audio.file_extensions().contains(&"wav"));

    assert!(Modality::Structured.file_extensions().contains(&"json"));
    assert!(Modality::Structured.file_extensions().contains(&"yaml"));

    assert!(Modality::Mixed.file_extensions().is_empty());
}

#[test]
fn test_modality_file_extensions_no_dots() {
    for modality in Modality::all() {
        for ext in modality.file_extensions() {
            assert!(
                !ext.starts_with('.'),
                "Extension '{}' should not start with dot",
                ext
            );
            assert_eq!(
                *ext,
                ext.to_lowercase(),
                "Extension '{}' should be lowercase",
                ext
            );
        }
    }
}

#[test]
fn test_modality_primary_embedding_model() {
    assert_eq!(Modality::Text.primary_embedding_model(), "E1_Semantic");
    assert_eq!(Modality::Code.primary_embedding_model(), "E7_Code");
    assert_eq!(Modality::Image.primary_embedding_model(), "E10_Multimodal");
    assert_eq!(Modality::Audio.primary_embedding_model(), "E10_Multimodal");
    assert_eq!(
        Modality::Structured.primary_embedding_model(),
        "E1_Semantic"
    );
    assert_eq!(Modality::Mixed.primary_embedding_model(), "E1_Semantic");
}

#[test]
fn test_modality_display() {
    assert_eq!(format!("{}", Modality::Text), "Text");
    assert_eq!(format!("{}", Modality::Code), "Code");
    assert_eq!(format!("{}", Modality::Image), "Image");
    assert_eq!(format!("{}", Modality::Audio), "Audio");
    assert_eq!(format!("{}", Modality::Structured), "Structured");
    assert_eq!(format!("{}", Modality::Mixed), "Mixed");
}

#[test]
fn test_modality_serde_roundtrip() {
    for modality in Modality::all() {
        let json = serde_json::to_string(&modality).expect("serialize failed");
        let parsed: Modality = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(modality, parsed, "Roundtrip failed for {:?}", modality);
    }
}

#[test]
fn test_modality_all_variants() {
    let all = Modality::all();
    assert_eq!(all.len(), 6);
    assert!(all.contains(&Modality::Text));
    assert!(all.contains(&Modality::Code));
    assert!(all.contains(&Modality::Image));
    assert!(all.contains(&Modality::Audio));
    assert!(all.contains(&Modality::Structured));
    assert!(all.contains(&Modality::Mixed));
}

#[test]
fn test_modality_clone_copy() {
    let original = Modality::Code;
    let cloned = original;
    let copied = original;
    assert_eq!(original, cloned);
    assert_eq!(original, copied);
}

#[test]
fn test_modality_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    for modality in Modality::all() {
        assert!(set.insert(modality), "Duplicate hash for {:?}", modality);
    }
    assert_eq!(set.len(), 6);
}

#[test]
fn test_modality_detect_empty_string() {
    // Empty string should return Text (default)
    assert_eq!(Modality::detect(""), Modality::Text);
    println!("BEFORE: empty string input");
    println!("AFTER: Modality::Text returned");
}

#[test]
fn test_modality_detect_whitespace_only() {
    assert_eq!(Modality::detect("   \n\t  "), Modality::Text);
    println!("BEFORE: whitespace-only input");
    println!("AFTER: Modality::Text returned");
}

#[test]
fn test_modality_detect_keyword_no_space() {
    // "function" without trailing space should NOT match
    // "functionality" is not code
    assert_eq!(Modality::detect("functionality test"), Modality::Text);
    println!("BEFORE: 'functionality test' (keyword embedded)");
    println!("AFTER: Modality::Text returned (not Code)");
}

//! Minimal test for tree-sitter-rust integration

use tree_sitter::Parser;

fn main() {
    let mut parser = Parser::new();

    // Try setting the language (0.23 API)
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    match parser.set_language(&language) {
        Ok(_) => println!("SUCCESS: Language set successfully"),
        Err(e) => println!("ERROR: Failed to set language: {:?}", e),
    }

    let code = "fn main() { println!(\"Hello\"); }";
    match parser.parse(code, None) {
        Some(tree) => {
            let root = tree.root_node();
            println!("PARSED: root node kind = {}", root.kind());
            println!("PARSED: has_error = {}", root.has_error());
            println!("PARSED: child_count = {}", root.child_count());
        }
        None => println!("ERROR: Failed to parse"),
    }
}

//! Terminal-safe normalization for externally sourced human-readable text.

/// Collapse a value to one terminal-safe line.
pub(crate) fn safe_inline(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() || is_directional_control(character) {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Preserve line structure while applying [`safe_inline`] to every line.
pub(crate) fn safe_multiline(value: &str) -> String {
    value
        .lines()
        .map(safe_inline)
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_directional_control(character: char) -> bool {
    matches!(
        character,
        '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_terminal_and_directional_controls() {
        let value = "safe\x1b]52;c;private\x07\u{202e}txt\nnext";
        let rendered = safe_inline(value);

        assert_eq!(rendered, "safe ]52;c;private txt next");
        assert!(!rendered.chars().any(char::is_control));
        assert!(!rendered.contains('\u{202e}'));
    }

    #[test]
    fn multiline_normalization_keeps_only_safe_line_boundaries() {
        assert_eq!(
            safe_multiline("one\x1b[2J\ntwo\rthree"),
            "one [2J\ntwo three"
        );
    }
}

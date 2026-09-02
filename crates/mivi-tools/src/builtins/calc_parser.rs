//! Pratt parser and expression tokenizer for arithmetic evaluation.

#[derive(Debug, PartialEq, Clone)]
pub enum MathToken {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

pub fn tokenize_expr(input: &str) -> Result<Vec<MathToken>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                chars.next();
            }
            '+' => {
                tokens.push(MathToken::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(MathToken::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(MathToken::Star);
                chars.next();
            }
            '/' => {
                tokens.push(MathToken::Slash);
                chars.next();
            }
            '^' => {
                tokens.push(MathToken::Caret);
                chars.next();
            }
            '(' => {
                tokens.push(MathToken::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(MathToken::RParen);
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num_str = String::new();
                let mut has_e = false;
                while let Some(&nc) = chars.peek() {
                    if nc.is_ascii_digit() || nc == '.' {
                        num_str.push(nc);
                        chars.next();
                    } else if (nc == 'e' || nc == 'E') && !has_e {
                        has_e = true;
                        num_str.push(nc);
                        chars.next();
                        if let Some(&sign) = chars.peek() {
                            if sign == '+' || sign == '-' {
                                num_str.push(sign);
                                chars.next();
                            }
                        }
                    } else {
                        break;
                    }
                }
                let val = num_str
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid number '{}': {}", num_str, e))?;
                tokens.push(MathToken::Number(val));
            }
            _ => return Err(format!("Unexpected character in math expression: '{}'", c)),
        }
    }
    Ok(tokens)
}

const MAX_PARSER_DEPTH: usize = 128;

pub struct PrattParser<'a> {
    tokens: &'a [MathToken],
    pos: usize,
    depth: usize,
}

impl<'a> PrattParser<'a> {
    pub fn new(tokens: &'a [MathToken]) -> Self {
        Self { tokens, pos: 0, depth: 0 }
    }

    pub fn peek(&self) -> Option<&MathToken> {
        self.tokens.get(self.pos)
    }

    pub fn next_token(&mut self) -> Option<&MathToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    pub fn infix_binding_power(op: &MathToken) -> Option<(u8, u8)> {
        match op {
            MathToken::Plus | MathToken::Minus => Some((1, 2)),
            MathToken::Star | MathToken::Slash => Some((3, 4)),
            MathToken::Caret => Some((6, 5)), // Right-associative exponentiation
            _ => None,
        }
    }

    pub fn prefix_binding_power(op: &MathToken) -> Option<u8> {
        match op {
            MathToken::Plus | MathToken::Minus => Some(7),
            _ => None,
        }
    }

    pub fn parse_expr(&mut self, min_bp: u8) -> Result<f64, String> {
        if self.depth >= MAX_PARSER_DEPTH {
            return Err("Expression nesting depth limit exceeded (max 128)".to_string());
        }
        self.depth += 1;
        let res = self.parse_expr_internal(min_bp);
        self.depth -= 1;
        res
    }

    fn parse_expr_internal(&mut self, min_bp: u8) -> Result<f64, String> {
        let mut lhs = match self.next_token() {
            Some(MathToken::Number(n)) => *n,
            Some(MathToken::Minus) => {
                let bp = Self::prefix_binding_power(&MathToken::Minus)
                    .ok_or_else(|| "Missing prefix binding power".to_string())?;
                let rhs = self.parse_expr(bp)?;
                -rhs
            }
            Some(MathToken::Plus) => {
                let bp = Self::prefix_binding_power(&MathToken::Plus)
                    .ok_or_else(|| "Missing prefix binding power".to_string())?;
                self.parse_expr(bp)?
            }
            Some(MathToken::LParen) => {
                let val = self.parse_expr(0)?;
                if self.next_token() != Some(&MathToken::RParen) {
                    return Err("Expected closing parenthesis ')'".to_string());
                }
                val
            }
            Some(tok) => return Err(format!("Unexpected token in prefix position: {:?}", tok)),
            None => return Err("Unexpected end of expression".to_string()),
        };

        while let Some(op) = self.peek() {
            if let Some((l_bp, r_bp)) = Self::infix_binding_power(op) {
                if l_bp < min_bp {
                    break;
                }
                let op = self
                    .next_token()
                    .cloned()
                    .ok_or_else(|| "Expected token".to_string())?;
                let rhs = self.parse_expr(r_bp)?;

                lhs = match op {
                    MathToken::Plus => lhs + rhs,
                    MathToken::Minus => lhs - rhs,
                    MathToken::Star => lhs * rhs,
                    MathToken::Slash => {
                        if rhs == 0.0 {
                            return Err("Division by zero".to_string());
                        }
                        lhs / rhs
                    }
                    MathToken::Caret => lhs.powf(rhs),
                    _ => unreachable!(),
                };
            } else {
                break;
            }
        }

        Ok(lhs)
    }
}

pub fn evaluate_expression(expr: &str) -> Result<f64, String> {
    let tokens = tokenize_expr(expr)?;
    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }
    let mut parser = PrattParser::new(&tokens);
    let res = parser.parse_expr(0)?;
    if parser.pos < tokens.len() {
        return Err("Unparsed trailing tokens in expression".to_string());
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_pratt_parser() {
        assert_eq!(evaluate_expression("3 + 4 * 2").unwrap(), 11.0);
        assert_eq!(evaluate_expression("(3 + 4) * 2").unwrap(), 14.0);
        assert_eq!(evaluate_expression("-5 + 10").unwrap(), 5.0);
        assert_eq!(evaluate_expression("3 - -5").unwrap(), 8.0);
        assert_eq!(evaluate_expression("-(3 * 2) + -4").unwrap(), -10.0);
        assert_eq!(evaluate_expression("100 / 4 / 5").unwrap(), 5.0);
    }

    #[test]
    fn test_calculator_recursion_depth_limit() {
        let mut deeply_nested = String::new();
        for _ in 0..150 {
            deeply_nested.push('(');
        }
        deeply_nested.push('1');
        for _ in 0..150 {
            deeply_nested.push(')');
        }
        let res = evaluate_expression(&deeply_nested);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("depth limit exceeded"));
    }

    #[test]
    fn test_calculator_scientific_notation() {
        assert_eq!(evaluate_expression("1e6 + 2e5").unwrap(), 1200000.0);
        assert_eq!(evaluate_expression("2.5e-3 * 1000").unwrap(), 2.5);
    }
}

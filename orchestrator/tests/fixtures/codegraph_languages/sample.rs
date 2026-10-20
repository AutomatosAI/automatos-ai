// Code-graph parser fixture (PRD-262 US-028): a struct, its impl with a method, a trait
// and a function.
pub struct Greeter {
    name: String,
}

impl Greeter {
    pub fn greet(&self) -> String {
        format!("Hello, {}", self.name)
    }
}

pub trait Polite {
    fn bow(&self);
}

fn make_greeter() -> Greeter {
    Greeter { name: String::from("world") }
}

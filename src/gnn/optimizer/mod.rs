use crate::gnn::*;
pub trait Optimizer<'a> {
    fn new<'b: 'a>(gnn: &'b mut GNN) -> Self;
    fn train(&mut self, dataset: &Vec<InputData>, batch_size: usize, epochs: usize);
}

pub mod sgd;
pub mod momentum_sgd;
pub mod adam;
pub use sgd::SGD;
pub use momentum_sgd::MomentumSGD;
pub use adam::Adam;

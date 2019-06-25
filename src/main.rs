extern crate ndarray;

mod gnn;

use ndarray::prelude::*;
use gnn::*;
use std::fs;
use std::env;

fn read_graph(n: u32) -> std::io::Result<String> {
    return fs::read_to_string("datasets/train/".to_string() + &n.to_string() + "_graph.txt");
}

fn read_label(n: u32) -> std::io::Result<String> {
    return fs::read_to_string("datasets/train/".to_string() + &n.to_string() + "_label.txt");
}

fn to_graph(input: &str) -> Graph {
    let mut adj_vec = Vec::new();
    let mut w = 0;
    let mut h = 0;
    for l in input.lines() {
        let mut v: Vec<i32> = l.split(" ").map(|x| x.parse().unwrap()).collect();
        if v.len() == 1 {
            continue;
        }
        w = v.len();
        adj_vec.append(&mut v);
        h += 1;
    }
    Array2::from_shape_vec((h, w), adj_vec).unwrap()
}

fn to_input_data(graph: &str, label: &str) -> InputData {
    InputData{
        graph: to_graph(graph),
        label: label.lines().next().unwrap().parse().unwrap()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1) {
        None => println!("Invalid param"),
        Some(s) => match s.as_str() {
            "task_01" => task_01(),
            "task_02" => task_02(),
            "task_03_sgd" => task_03(false),
            "task_03_msgd" => task_03(true),
            "task_04" => task_04(false),
            "task_04_test" => task_04(true),
            _ => println!("Invalid param"),
        }
    }
}

fn task_01() {
    let graph = arr2(&[
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]);
    let graph_weight = arr2(&[
        [1.0, 3.0, -2.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [2.0, 3.0, -1.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [2.0, 3.0, -1.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [1.0, 0.2, 2.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [1.0, 0.2, 2.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [2.0, 3.0, -1.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [1.0, 0.2, 2.0, 0.5, -0.5, 0.5, -0.5, 0.3],
        [1.0, 0.2, 2.0, 0.5, -0.5, 0.5, -0.5, 0.3]
    ]);
    let feature_vector = vec![
        arr1(&[-0.92, -0.32, -0.68, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[-0.68, -0.97, -0.46, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[-0.62, -0.35, -0.15, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[0.51, -0.04, 0.01, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[-0.91, -0.18, 0.04, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[-0.06, -0.88, -0.89, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[0.32, -0.50, 0.52, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[0.18, -0.40, 0.09, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[-0.77, 0.87, 0.16, -0.12, 0.13, 0.44, 0.38, -0.05]),
        arr1(&[0.64, -0.84, -0.61, -0.12, 0.13, 0.44, 0.38, -0.05])
    ];
    let output_weight = arr1(&[0.5, -0.3, 0.1, 0.2, -0.1, 0.3, -0.5, 0.3]);

    let mut gnn = GNN {
        graph: graph,
        graph_weight: graph_weight,
        feature_vector: feature_vector,
        output_weight: output_weight,
        f: relu,
        ..Default::default()
    };

    gnn.aggregate(2);
    println!("{:?}", gnn.readout());

}

fn task_02() {
    let graph = arr2(&[
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]);
    let graph_weight = get_square_matrix(8, 0.0, 0.4);
    let output_weight = get_vector(8, 0.0, 0.4);

    let mut gnn = GNN {
        graph: graph,
        graph_weight: graph_weight,
        output_weight: output_weight,
        alpha: 0.005,
        f: relu,
        ..Default::default()
    };

    gnn.init_vector();
    gnn.aggregate(2);
    println!("Loss: {}", gnn.loss_function(1));
    for _j in 1..5000 {
        gnn.train(1);
        let loss = gnn.loss_function(1);
        println!("Loss: {}", loss);
        if loss < 0.01 {
            break;
        }
    }

}

fn task_03(is_momentum: bool) {
    let graph_weight = get_square_matrix(8, 0.0, 0.4);
    let output_weight = get_vector(8, 0.0, 0.4);

    let mut data_num = 0;
    let mut dataset: Vec<InputData> = Vec::new();
    loop {
        if data_num >= 1600 {
            break;
        }
        let graph = match read_graph(data_num) {
            Ok(s) => s,
            _ => break
        };
        let label = match read_label(data_num) {
            Ok(s) => s,
            _ => break
        };
        let input_data = to_input_data(&graph, &label);
        dataset.push(input_data);
        data_num += 1;
    }

    let mut test_dataset: Vec<InputData> = Vec::new();
    loop {
        let graph = match read_graph(data_num) {
            Ok(s) => s,
            _ => break
        };
        let label = match read_label(data_num) {
            Ok(s) => s,
            _ => break
        };
        let input_data = to_input_data(&graph, &label);
        test_dataset.push(input_data);
        data_num += 1;
    }

    let mut gnn = GNN {
        graph_weight: graph_weight,
        output_weight: output_weight,
        output_bias: 0.0,
        epsilon: 0.001,
        alpha: 0.005,
        moment: 0.9,
        f: relu,
        ..Default::default()
    };
    {
        if is_momentum {
            let mut optimizer = optimizer::MomentumSGD::new(&mut gnn);
            optimizer.train(&dataset, 320, 100);
        } else {
            let mut optimizer = optimizer::SGD::new(&mut gnn);
            optimizer.train(&dataset, 320, 100);
        }
    }
    let train_accuracy = gnn.test_accuracy(&dataset);
    println!("Train Accuracy: {}", train_accuracy);
    let accuracy = gnn.test_accuracy(&test_dataset);
    let average_loss = gnn.test_average_loss(&test_dataset);
    println!("Test Accuracy: {}, Average Loss: {}", accuracy, average_loss);
}

fn task_04(predict: bool) {
    let graph_weight = get_square_matrix(8, 0.0, 0.4);
    let output_weight = arr1(&[0.5, -0.3, 0.1, 0.2, -0.1, 0.3, -0.5, 0.3]);

    let mut data_num = 0;
    let mut dataset: Vec<InputData> = Vec::new();
    loop {
        if !predict && data_num >= 1600 {
            break;
        }
        let graph = match read_graph(data_num) {
            Ok(s) => s,
            _ => break
        };
        let label = match read_label(data_num) {
            Ok(s) => s,
            _ => break
        };
        let input_data = to_input_data(&graph, &label);
        dataset.push(input_data);
        data_num += 1;
    }

    if !predict {
        let mut test_dataset: Vec<InputData> = Vec::new();
        loop {
            let graph = match read_graph(data_num) {
                Ok(s) => s,
                _ => break
            };
            let label = match read_label(data_num) {
                Ok(s) => s,
                _ => break
            };
            let input_data = to_input_data(&graph, &label);
            test_dataset.push(input_data);
            data_num += 1;
        }

        let mut gnn = GNN {
            graph_weight: graph_weight,
            output_weight: output_weight,
            output_bias: 0.0,
            epsilon: 0.001,
            alpha: 0.005,
            moment: 0.9,
            f: relu,
            ..Default::default()
        };
        {
            let mut optimizer = optimizer::Adam::new(&mut gnn);
            optimizer.train(&dataset, 320, 100);
        }
        let train_accuracy = gnn.test_accuracy(&dataset);
        println!("Train Accuracy: {}", train_accuracy);
        let accuracy = gnn.test_accuracy(&test_dataset);
        let average_loss = gnn.test_average_loss(&test_dataset);
        println!("Test Accuracy: {}, Average Loss: {}", accuracy, average_loss);
    } else {
         let mut gnn = GNN {
            graph_weight: graph_weight,
            output_weight: output_weight,
            output_bias: 0.0,
            epsilon: 0.001,
            alpha: 0.005,
            moment: 0.9,
            f: relu,
            ..Default::default()
        };
        {
            let mut optimizer = optimizer::Adam::new(&mut gnn);
            optimizer.train(&dataset, 400, 100);
        }
        let mut test_data_num = 0;
        let mut tests: Vec<Graph> = Vec::new();
        loop {
            let graph = match read_graph(test_data_num) {
                Ok(s) => s,
                _ => break
            };
            tests.push(to_graph(&graph));
            test_data_num += 1;
        }
        println!("------------------------------------");
        for t in tests.iter() {
            let output = gnn.classify(t);
            println!("{}", output);
        }
    }
}

// Tests
#[allow(dead_code)]
fn get_gnn_for_test() -> GNN {
    let graph = arr2(&[
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]);
    let graph_weight = arr2(&[
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0]
    ]);
    let feature_vector = vec![
        arr1(&[1.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0])
    ];

    let output_weight = arr1(&[1.0, 1.0, 1.0]);

    GNN {
        graph: graph,
        graph_weight: graph_weight,
        feature_vector: feature_vector,
        output_weight: output_weight,
        f: relu,
        ..Default::default()
    }
}

#[test]
fn create_gnn() {
    get_gnn_for_test();
}

#[test]
fn aggregate() {
    let mut gnn = get_gnn_for_test();
    gnn.aggregate(1);
    let sample_result = vec![
        arr1(&[1.0, 0.0, 1.0]),
        arr1(&[2.0, 0.0, 2.0]),
        arr1(&[1.0, 0.0, 1.0])
    ];
    assert_eq!(gnn.feature_vector, sample_result);
}

#[test]
fn readout() {
    let mut gnn = get_gnn_for_test();
    gnn.aggregate(2);
    let h_g = gnn.readout();
    let sample_result = arr1(&[12.0, 12.0, 6.0]);
    assert_eq!(h_g, sample_result);
}

#[test]
fn weighted_sum() {
    let mut gnn = get_gnn_for_test();
    gnn.aggregate(2);
    let weighted_sum = gnn.weighted_sum();
    let sample_result = 30.0;
    assert_eq!(weighted_sum, sample_result);
}

#[test]
fn init_vector() {
    let mut gnn = get_gnn_for_test();
    gnn.init_vector();
    let sample_result = vec![
        arr1(&[1.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0])
    ];
    assert_eq!(gnn.feature_vector, sample_result);
}


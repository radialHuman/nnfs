// https://www.youtube.com/watch?v=tMrbN67U9d4

// Dot product

pub fn three() {
    println!("Shapes",);
    let v1 = vec![1, 2, 3, 4, 5];
    println!("The shape of {:?} is {} in one dimension", v1, v1.len());
    let v2 = vec![[1, 2, 3, 4, 5], [4, 3, 6, 1, 4]]; // (row,columns)
    println!(
        "The shape of {:?} is ({},{}) in {} dimension",
        v2,
        v2.len(),
        v2[0].len(),
        v2.len()
    );
}

/*
OUTPUT
Shapes
The shape of [1, 2, 3, 4, 5] is 5 in one dimension
The shape of [[1, 2, 3, 4, 5], [4, 3, 6, 1, 4]] is (2,5) in 2 dimension
*/

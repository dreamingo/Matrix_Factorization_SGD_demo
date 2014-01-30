Matrix Factorization Demo For Collaborative Filtering
====================================================

A demo of Bias-Matrix-Factorization for collaborative filtering, sovled with SGD(stochastic gradient descent)

### Data set:

    This demo use the dataset of [MovieLens 1M](http://grouplens.org/datasets/movielens/)

### Directory Structure

    .
    ├── ml-1m
    │   ├── new_ratings.dat
    │   ├── ratings_new.dat
    ├── README
    ├── mf.cpp

### Usage

    g++ -g mf.cpp -o mf 
    ./mf

    
### Performance

    Use python to random shuffle the ratings, and use 4/5 of the record as trainning set and the remain as a validation set;
    The best performance with RMSE 0.8544

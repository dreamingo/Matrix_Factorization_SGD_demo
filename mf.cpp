// Data Structure:
//     rating Matrix: r
//         We use a sparse storage for the rating Matrix r:
//              int* r: rating matrix of len m(m == num_of_ratings)
//              int* col_index: column index of the rating 
//              int* row_index: row index of the rating 
//         Here we don't use a data structure for the rating matrix storage. 
//         We read it from file before we use everytime.(For the sake of memory save)
//     User perference:    double** q;
//     Movie properties:   double** p; 
//     User bias array:    double* b_u;
//     item bias array:    double* b_i;
// 
// Parameter:
//     int K:          The number of the latent factor;
//     double l_rate:     Learning rate for the gradient descent;
//     double lamdba:     The factor for regularization;
//     double sum_g_q:      Sum of the gradient of q
//     double sum_g_p:      Sum of the gradient of p
// 
// Formulation:
//     r_{ui} = q^{t}_{u}.p{i} + u + b_{i} + b_{u} + lamdba(q^2 + p^2 + b_u^2 + b_i^2)
// 
// Update Formulation:
//     User perference:    q_i += l_rate{ (r_{ui} - b_u - b_i - p^t.q_i}).p_u - lambda.q_i}
//     Item perference:    p_i += l_rate{ (r_{ui} - b_u - b_i - p^t.q_i}).q_i - lambda.p_u}
// 
// Stop Criterion:
//     double tol:     Tolerance for stopping criterion
//     int max_iter:   Maximum iteration for the algorithm
//     if (sum of the gradient < tol || num_iteration > max_iter)
//         stop...
    
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>

#define Print_err(s) {fprintf(stderr, "Error:%s\n", s); exit(-1);}
// #define BIAS_EFFECTIVE true
#define BIAS_EFFECTIVE true 

// const char* file_name = "ml-1m/ratings.dat";
const char* file_name = "ml-1m/new_ratings.dat";
const int user_num = 6040;          /* The number of user in the rating file */
const int movie_num = 3952;         /* The number of movie in the rating file */
const int record_num = 1000209;     /* The number of the ratings*/
int train_num;

const int K = 40;                   /* The number of the latent factor*/
const int max_iter = 100;           /* Maximum iteration*/
const double l_rate = 0.002;         /* learning rate*/
const double lamdba = 0.06;         /* Parameter for regularization*/
const double bias_reg = 0.002;
const double tol = 0.1;             /* Tolerance for the stop criterion*/
double average;

int *r;                             /* Rating array*/
int *col_index;                     /* The column index of the rating*/
int *row_index;                     /* The row index of the ratings*/
double **q;                         /* The q(user perference) matrix*/
double **p;                         /* The p(item properties) matrix*/
double *b_i;                        /* The item bias array*/
double *b_u;                        /*The user bias array*/

//Use for debug;
void print_array(const double* a){
    // printf("-------------------------------------\n");
    for(int i = 0; i < K; i++) printf("%.5f, ", a[i]);
    printf("\n");
}

double InnerProduct(const double* a, const double* b){
    double sum = 0;
    for(int i = 0; i < K; i++) sum += a[i]*b[i];
    return sum;
}


void LoadData(const char* file_name){
    FILE* fptr = fopen(file_name, "r");
    if(fptr == NULL)
        Print_err("Fail in loading data!\n");
    r = (int*)malloc(sizeof(int)*(record_num+1));
    col_index = (int*)malloc(sizeof(int)*(record_num+1));
    row_index = (int*)malloc(sizeof(int)*(record_num+1));
    int date, i = 0;
    while(fscanf(fptr, "%d::%d::%d::%d", &row_index[i],
                &col_index[i], &r[i], &date) !=EOF)  i++;
    printf("Successfully load data! There are totally %d records\n", i);
}

void InitData(){
    train_num = record_num * 0.8;           /* The number for training*/
    double sum = 0;
    srand48(time(NULL));
    for(int i = 0; i < train_num; i++) sum += r[i];
    average = sum/train_num;

    q = (double**)malloc(user_num * sizeof(double*));
    for(int i = 0; i < user_num; i++){
        q[i] = (double*)malloc(K * sizeof(double));
        // memset(q[i], 0, sizeof(double)*K);
        for(int j = 0; j < K; j++)
            q[i][j] = drand48();
            // q[i][j] = 0.1;
    }

    p = (double**)malloc(movie_num * sizeof(double*));
    for(int i = 0; i < movie_num; i++){
        p[i] = (double*)malloc(K * sizeof(double));
        // memset(p[i], 0, sizeof(double)*K);
        for(int j = 0; j < K; j++)
            p[i][j] = drand48();
            // p[i][j] = 0.1;
    }

    b_i = (double*)malloc(movie_num * sizeof(double));
    b_u = (double*)malloc(user_num * sizeof(double));

    for(int i = 0; i < movie_num; i++) b_i[i] = drand48()-0.5;
    for(int i = 0; i < user_num; i++) b_u[i] = drand48()-0.5;

    // for(int i = 0; i < movie_num; i++) b_i[i] = drand48() - 0.5;
    // for(int i = 0; i < user_num; i++) b_u[i] = drand48() - 0.5;
}

double Predict(const int index){
    //row means for user index;
    int row = row_index[index] - 1;
    //col means for movie index;
    int col = col_index[index] - 1;
    double r_predict = average + b_u[row] + b_i[col];
    for(int i = 0; i < K; i++)
        r_predict += q[row][i] * p[col][i];
    return r_predict;
}

double TrainRMSE(){
    double sum = 0.0, err, r_predict;
    for(int i = 0; i < train_num; i++){
        r_predict = Predict(i);
        if(r_predict > 5) r_predict = 5;
        if(r_predict < 1) r_predict = 1;
        err = r_predict - r[i];
        sum += err * err;
    }
    // printf("TrainRMSE:sum:%.2f\n", sum);
    return sqrt(sum/train_num);
}

double ValidationRMSE(){
    double sum = 0.0, err,r_predict;
    for(int i = train_num; i < record_num; i++){
        r_predict = Predict(i);
        if(r_predict - 5 > 0) r_predict = 5;
        if(r_predict - 1 < 0) r_predict = 1;
        err = r_predict - r[i];
        sum += err * err;
    }
    // printf("ValidationRMSE:sum:%.2f\n", sum);
    return sqrt(sum/(record_num - train_num));

}

void Update(const int& i){
    int user_index = row_index[i] - 1;
    int movie_index = col_index[i] - 1;

    double inner_product = InnerProduct(q[user_index], p[movie_index]);
    double err = r[i] - average - b_u[user_index] - b_i[movie_index] 
        - inner_product;
    // printf(".................................\n");
    // printf("Err:%.5f\n", err);
    // printf("r[%d]:%d\n",i, r[i]);
    // printf("b_u[%d]:%.5f\n",user_index, b_u[user_index]);
    // printf("b_i[%d]:%.5f\n",movie_index, b_i[movie_index]);
    // printf("q[%d]:", user_index);
    // print_array(q[user_index]);
    // printf("p[%d]:", movie_index);
    // print_array(p[movie_index]);
    // printf("InnerProduct: %.5f\n", inner_product);
    //
    // Update for q_u array
    // q_u +=  l_rate(err.p_i - \lamdba.q_u)
    for(int i = 0; i < K; i++)
        q[user_index][i] += l_rate * (err*p[movie_index][i]  - lamdba * q[user_index][i]);
    // Update for p_i array
    // p_i +=  l_rate(err.q_u - \lamdba.p_i)
    for(int i = 0; i < K; i++)
        p[movie_index][i] += l_rate * (err*q[user_index][i] - lamdba * p[movie_index][i]);

    if(BIAS_EFFECTIVE){
        //Update for b_u
        b_u[user_index] += l_rate * (err - bias_reg * b_u[user_index]);
        //Update for b_i
        b_i[movie_index] += l_rate * (err - bias_reg * b_i[movie_index]);
    }
}

void Train(){
    printf("TrainRMSE: %.5f\n", TrainRMSE());
    printf("ValidationRMSE: %.5f\n", ValidationRMSE());
    for(int iter = 0; iter < max_iter; iter++){
        for(int i = 0; i < train_num; i++)
            Update(i);
        printf("---------- The %d iteration -----------------\n", iter);
        printf("TrainRMSE: %.5f\n", TrainRMSE());
        printf("ValidationRMSE: %.5f\n", ValidationRMSE());
    }
}

void  DebugInfo(){
    for(int i = 0; i < user_num; i++)
        print_array(q[i]);
    printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");

    for(int i = 0; i < movie_num; i++)
        print_array(p[i]);
}

int main(int argc, char** argv){
    time_t begin = clock();
    LoadData(file_name);
    InitData();
    time_t loadTime = clock();
    printf("LoadTime: %.3f\n", double((loadTime - begin))/CLOCKS_PER_SEC);

    Train();
    // DebugInfo();
    time_t end = clock();
    printf("End: %.3f\n", double((end - begin))/CLOCKS_PER_SEC);
}

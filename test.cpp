/**
  Direct Odometry from RGB-D Images
  Autor: Max W. Portocarrero
  Compilamos con el siguiente comando
  g++ test.cpp -o test `pkg-config opencv --cflags --libs` && ./test
**/

// Import Standart Libraries
#include <iostream>

#include "utilities.hpp"
#include "linear_algebra_functions.hpp"

enum image_type {intensity, depth, intrinsics};
cv::Mat downscale(const cv::Mat & image, int level, int type);

void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd xi, const cv::Mat K);


int main(){

    // usando opencv
    // El algoritmo que usa para pasar a grayscale es distintoal de Matlab
    cv::Mat i0,i1;
    if(read_image(i0,"test_data/rgb/test1_i0.png"))
        cv::imshow("Display Image i0",i0);
    if(read_image(i1,"test_data/rgb/test1_i1.png"))
        cv::imshow("Display Image i1",i1);

    cv::Mat d0;
    if(read_depth_image(d0,"test_data/depth/test1_d0.png",5000.0f))
        show_depth_image("depth at t0",d0);

    // Parámetros intrinsecos para estas imágenes
    float k[3][3] = {{517.3f,   0.0f, 318.6f},
                     {  0.0f, 516.5f, 255.3f},
                     {  0.0f,   0.0f,   1.0f}};
    //Expresados en cv::Mat
    cv::Mat K = cv::Mat(cv::Size(3,3),CV_32F,k);

    // proceso de downscale
    cv::Mat i0_scaled, i1_scaled, d0_scaled,K_scaled;
    i0_scaled = downscale(i0,0,intensity);
    i1_scaled = downscale(i1,0,intensity);
    d0_scaled = downscale(d0,0,depth);
    K_scaled = downscale(K,0,intrinsics);

/*
    cv::namedWindow("Scaled Depth",cv::WINDOW_AUTOSIZE);
    show_depth_image("Scaled Depth",d0_scaled);

    //std::cout << "image = " << d0_scaled << std::endl; // Para visualizar imagen
    std::cout << "K = " << K << std::endl;
    std::cout << "Ks = " << K_scaled << std::endl;
*/
    Eigen::VectorXd xi(6);
    //xi << 0,0,0,0,0,0;
    //xi << 1,2,3,4,5,6;
    xi << -0.0018, 0.0065, 0.0369, -0.0287, -0.0184, -0.0004; // real

    CalcDiffImage(i0_scaled,d0_scaled,i1_scaled,xi,K_scaled);

    cv::waitKey(0);

}

cv::Mat downscale(const cv::Mat & image, int level, int type){
    if(level == 0) return image;

    switch(type){
        case intensity:{
            int rows = image.rows;
            int cols = image.cols;

            // creamos una matriz que calcula la mitad de la imagen
            cv::Mat scaled_image = cv::Mat::zeros(cv::Size(cols/2,rows/2),image.type());

            //std::cout << scaled_image.size << std::endl;

            for(int j = 0; j < rows/2; j++){
                for(int i = 0; i < cols/2; i++){
                    scaled_image.at<myNum>(j,i) = image.at<myNum>(2*j,2*i) +
                                                   image.at<myNum>(2*j+1,2*i) +
                                                   image.at<myNum>(2*j,2*i+1) +
                                                   image.at<myNum>(2*j+1,2*i+1);
                    scaled_image.at<myNum>(j,i) /= 4;

                    //std::cout << image.at<float>(j,i) << " "; // para imprimir los datos
                 }
                    //std::cout << std::endl;
             }

            return downscale(scaled_image,level -1,type);
            break;
        }
        case depth:{

            int rows = image.rows;
            int cols = image.cols;

            // creamos una matriz que calcula la mitad de la imagen
            cv::Mat scaled_image = cv::Mat::zeros(cv::Size(cols/2,rows/2),image.type());

            for(int j = 0; j < rows/2; j++){
                for(int i = 0; i < cols/2; i++){
                    int cont = 0;
                    //Contamos la cantidad de pixeles no nulos en la vecindad del pixel
                    if(image.at<myNum>(2*j,2*i))
                        cont++;
                    if(image.at<myNum>(2*j+1,2*i))
                        cont++;
                    if(image.at<myNum>(2*j,2*i+1))
                        cont++;
                    if(image.at<myNum>(2*j+1,2*i+1))
                        cont++;

                    scaled_image.at<myNum>(j,i) = image.at<myNum>(2*j,2*i) +
                                                   image.at<myNum>(2*j+1,2*i) +
                                                   image.at<myNum>(2*j,2*i+1) +
                                                   image.at<myNum>(2*j+1,2*i+1);

                    if(cont == 0.0f) scaled_image.at<myNum>(j,i) = 0.0f;
                    else scaled_image.at<myNum>(j,i) /= cont;
                    //std::cout  << cont << " ";
                    //std::cout <<scaled_image.at<myNum>(j,i) << " "; // para imprimir los datos
                }
                 //std::cout << std::endl;
            }
            return downscale(scaled_image,level -1,type);
            break;
        }
        // En el caso de la matriz intrinseca es una matrix 3 x 3
        case intrinsics:{
            cv::Mat scaled_intrinsics = cv::Mat::eye(cv::Size(3,3),image.type());

            scaled_intrinsics.at<float>(0,0) = image.at<float>(0,0) / 2.0f;
            scaled_intrinsics.at<float>(1,1) = image.at<float>(1,1) / 2.0f;
            scaled_intrinsics.at<float>(0,2) = (image.at<float>(0,2) + 0.5f) / 2.0f - 0.5f;
            scaled_intrinsics.at<float>(1,2) = (image.at<float>(1,2) + 0.5f) / 2.0f - 0.5f;

            return downscale(scaled_intrinsics,level -1, type);
            break;
        }
        default:{
            break;
        }
    }




}



// Funcion que calcula los residuales entre imágenes
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd xi, const cv::Mat K){
    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;
    // Declaramos una matriz para guardar los residuales
    cv::Mat mat_residuals = cv::Mat::zeros(cv::Size(cols,rows),CV_32FC1);
    // Calculamos la transformación rigid-body motion
    Eigen::Matrix4d g = twistcoord2rbm(xi);

    // Pasamos la matriz intrinseca a Eigen
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<myNum>(0,0), K.at<myNum>(0,1), K.at<myNum>(0,2),
               K.at<myNum>(1,0), K.at<myNum>(1,1), K.at<myNum>(1,2),
               K.at<myNum>(2,0), K.at<myNum>(2,1), K.at<myNum>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();

    // Calculamos las nuevas coordenadas
    FOR(j,rows)
        FOR(i,cols){
            if( d0.at<myNum>(j,i) > 0 ){
                Eigen::Vector2d warped_coord;
                Eigen::Vector2d coord0(i,j);
                //std::cout << "x,y " << coord0 << " ;";

                Eigen::Vector3d world_coord;
                world_coord << coord0 , 1;
                world_coord = eigen_K_inverse * d0.at<myNum>(j,i) * world_coord;
                //std::cout << world_coord << " ;";

                // Transformed coord by rigid-body motion
                Eigen::Vector4d transformed_coord;
                transformed_coord << world_coord, 1;
                transformed_coord = g * transformed_coord;
                //std::cout << transformed_coord << " ;";

                Eigen::Vector3d projected_coord;
                projected_coord << transformed_coord(0), transformed_coord(1), transformed_coord(2);
                projected_coord = eigen_K * projected_coord;
                //std::cout << projected_coord << " ;";

                warped_coord << projected_coord(0) / projected_coord(2), projected_coord(1) / projected_coord(2);
                //std::cout << warped_coord << " ;\n";


                // Revisamos si las coord se encuentran dentro de los limites de la imagen
                if( warped_coord(0) > 0 && warped_coord(0) < cols - 1 &&
                    warped_coord(1) > 0 && warped_coord(1) < rows - 1 ){
                    float a = warped_coord(1) - floor(warped_coord(1));
                    float b = warped_coord(0) - floor(warped_coord(0));
                    int t = floor(warped_coord(1)), d = ceil(warped_coord(1));
                    int s = floor(warped_coord(0)), r = ceil(warped_coord(0));

                    myNum t_s = i1.at<myNum>(t,s);
                    myNum d_s = i1.at<myNum>(d,s);
                    myNum t_r = i1.at<myNum>(t,r);
                    myNum d_r = i1.at<myNum>(d,r);

                    myNum result = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);
                    //std::cout << result << " ";

                    mat_residuals.at<myNum>(j,i) = result;
                }
            } // Fin de Condicional exterior
        } // Fin Bucle FOR

    cv::Mat diff = mat_residuals - i0; // Revisar estas operacioens para el calculo de los maps!!!!

    FOR(j,rows)
        FOR(i,cols){
            if(mat_residuals.at<myNum>(j,i) == 0){
                diff.at<myNum>(j,i) = -1;
            }
        }
    // Como las diferencias entre imágenes están en el rango de [-1,1]
    // Sumamos 1 a todos los valores para que el intervalo vaya de [0,2]
    diff = diff + 1.0f;

    // Aquí aplicaremos un mapeo proporcional a este intervalo
    // y le aplicamos una mascara de colores para observar las zonas
    // de mayor diferencia

    double min,max;
    cv::minMaxIdx(diff, &min, &max);
    std::cout << "max: " << max << "min: " << min << std::endl;

    cv::Mat adjMap;
    cv::convertScaleAbs(diff, adjMap, 255 / max);

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff", FalseColorMap);

    /** // Showing dual difference color maps
    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff2", FalseColorMap2);
    **/
}


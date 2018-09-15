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

void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, Eigen::VectorXd & xi, const cv::Mat K);
// Estas funciones me ayudaran a calcular los gradientes en direccion X e Y para una imagen de entrada
void Gradient(const cv::Mat & InputImg, cv::Mat & OutputImg, cv::Mat &OutputYImg);

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

/*
    cv::namedWindow("Scaled Depth",cv::WINDOW_AUTOSIZE);
    show_depth_image("Scaled Depth",d0_scaled);

    //std::cout << "image = " << d0_scaled << std::endl; // Para visualizar imagen
    std::cout << "K = " << K << std::endl;
    std::cout << "Ks = " << K_scaled << std::endl;
*/
    Eigen::VectorXd xi(6);
    xi << 0,0,0,0,0,0;
    //xi << 1,2,3,4,5,6;
    //xi << -0.0018, 0.0065, 0.0369, -0.0287, -0.0184, -0.0004; // Aproximacion


    for(int lvl = 4; lvl >= 4; --lvl){
        std::cout << std::endl << "level = " << lvl << std::endl << std::endl;

        // proceso de downscale
        cv::Mat i0_scaled, i1_scaled, d0_scaled,K_scaled;
        i0_scaled = downscale(i0,lvl,intensity);
        i1_scaled = downscale(i1,lvl,intensity);
        d0_scaled = downscale(d0,lvl,depth);
        K_scaled = downscale(K,lvl,intrinsics);

        // Actualizamos el valor de xi
        CalcDiffImage(i0_scaled,d0_scaled,i1_scaled,xi,K_scaled);

    }

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
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, Eigen::VectorXd &xi, const cv::Mat K){
    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;

    // Pasamos la matriz intrinseca a Eigen+
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<myNum>(0,0), K.at<myNum>(0,1), K.at<myNum>(0,2),
               K.at<myNum>(1,0), K.at<myNum>(1,1), K.at<myNum>(1,2),
               K.at<myNum>(2,0), K.at<myNum>(2,1), K.at<myNum>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();
    double fx = eigen_K(0,0), fy = eigen_K(1,1);
    std::cout << "input xi=" << xi.transpose() << std::endl;

    double last_err = 10000000.0; // Vamos minimizando el error

    /** Inicio de las interaciones **/
    FOR(it,40){
        std::cout << "* iteracion " << it << std::endl;
                // Declaramos una matriz para guardar los residuales
                cv::Mat i1_warped = cv::Mat::zeros(cv::Size(cols,rows),CV_32FC1);
                // Calculamos la transformación rigid-body motion
                Eigen::Matrix4d g = twistcoord2rbm(xi);

                // Creamos nuestros mapeos para x e y
                cv::Mat map_warped_x, map_warped_y;
                map_warped_x.create(i1.size(), CV_32FC1);
                map_warped_y.create(i1.size(), CV_32FC1);

                // Y los mapeos para los Warp Coordinates(no proyectados)
                // restamos 100 para simular NaN values
                cv::Mat xp, yp, zp;
                xp = cv::Mat::zeros(i1.size(),CV_32FC1) - 100;
                yp = cv::Mat::zeros(i1.size(),CV_32FC1) - 100;
                zp = cv::Mat::zeros(i1.size(),CV_32FC1) - 100;

                // Calculamos las nuevas coordenadas
                FOR(j,rows)
                    FOR(i,cols){
                        if( d0.at<myNum>(j,i) > 0 ){
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

                            Eigen::Vector2d warped_coord;
                            warped_coord << projected_coord(0) / projected_coord(2), projected_coord(1) / projected_coord(2);
                            //std::cout << warped_coord << " ;\n";

                            // Probemos usar los mapeos de opencv
                            map_warped_x.at<myNum>(j,i) = warped_coord(0);
                            map_warped_y.at<myNum>(j,i) = warped_coord(1);

                            // Verificamos que el número sea positivo
                            // Para que no haya problema al calcular el Jacobiano
                            if(transformed_coord(2) > 0.0f){
                                xp.at<myNum>(j,i) = transformed_coord(0);
                                yp.at<myNum>(j,i) = transformed_coord(1);
                                zp.at<myNum>(j,i) = transformed_coord(2);
                            }

                            /**
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
                            **/
                        } else {
                            map_warped_x.at<myNum>(j,i) = -100;
                            map_warped_y.at<myNum>(j,i) = -100;
                        } // Fin de Condicional exterior
                    } // Fin Bucle FOR

                // Calculo de la gradiente
                cv::Mat XGradient, YGradient;
                Gradient(i1,XGradient,YGradient);

                // Interpolamos los valores para los warped coordinates
                cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));
                cv::Mat residuals = cv::Mat::zeros(i1.size(),CV_32FC1);
                residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!

                // Interpolamos sobre las gradientes
                cv::Mat map_XGradient, map_YGradient;
                //map_XGradient.create(i1.size(), i1.type());
                //map_YGradient.create(i1.size(), i1.type());
                cv::remap(XGradient,map_XGradient,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));
                cv::remap(YGradient,map_YGradient,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));


                /*** CONSTRUCION DE c y r0 ***/
                // Formamos las matrices pero usando la libreria eigen
                Eigen::VectorXd r(rows * cols);
                Eigen::MatrixXd c(rows * cols,6);
                int cont = 0;
                FOR(j,rows){
                    FOR(i,cols){
                        // Sólo usaremos los datos que sean válidos
                        // Es decir aquellos que tengan valores válidos de gradiente
                        // 1 < map_warped_x < width -1; 1 < map_warped_y < height -1
                        // Valores válidos para el Image warped
                        // i1_warped != 0
                        // Valores válidos para las coordenadas de pixel en 3D
                        // xp,yp,zp != -100
                        if( (1 < map_warped_x.at<myNum>(j,i) && map_warped_x.at<myNum>(j,i) < cols-1) &&
                            (1 < map_warped_y.at<myNum>(j,i) && map_warped_y.at<myNum>(j,i) < rows-1) &&
                            i1_warped.at<myNum>(j,i) != 0 &&
                            xp.at<myNum>(j,i) != -100){

                            // Residuales
                            r(cont) = residuals.at<myNum>(j,i);


                            double gradX = XGradient.at<myNum>(j,i), gradY = YGradient.at<myNum>(j,i);
                            gradX *= fx; gradY *= fy;
                            double x = xp.at<myNum>(j,i), y = yp.at<myNum>(j,i), z = zp.at<myNum>(j,i);

                            // Jacobiano
                            c(cont,0) = gradX / z;
                            c(cont,1) = gradY / z;
                            c(cont,2) = -( gradX * x + gradY * y ) / (z*z);
                            c(cont,3) = -( gradX * x * y / (z*z)) -  (gradY * (1 + (y*y)/(z*z)));
                            c(cont,4) = ( gradX * (1 + (x*x)/(z*z))) + (gradY * x * y / (z*z));
                            c(cont,5) = (- gradX * y + gradY * x) / z;


                            cont++;
                        }
                    }
                }

                std::cout << "cont: <<" << cont << std::endl;

                // Hacemos un slice con el conteo de "cont" pixeles validos
                Eigen::VectorXd R0 = r.block(0,0,cont,1);
                Eigen::MatrixXd J = c.block(0,0,cont,6); J = -J;

                Eigen::MatrixXd J_inv = (J.transpose() * J);

                // Calculamos nuestro diferencial de xi
                Eigen::VectorXd d_xi = -J_inv.inverse() * J.transpose() * R0;

                std::cout << "d_xi:\n" << d_xi.transpose() << std::endl;

                xi = rbm2twistcoord( twistcoord2rbm(d_xi) * g);

                std::cout << "xi:\n" << xi.transpose() << std::endl;

                double err = R0.dot(R0);
                std::cout << "err=" << err << " last_err=" << last_err << std::endl;

                //if( err / last_err > 0.995){
                //    return;
                //}

                last_err = err;


                /*FILTRAMOS LOS VALORES CON VALOR 0 DESPUES DE LA INTERPOLACION*/
                FOR(j,rows)
                    FOR(i,cols){
                        // la siguiente condición funciona por la mayoria
                        // de los numeros que si son tomados en cuenta no son exactamente 0
                        if(i1_warped.at<myNum>(j,i) == 0){
                            residuals.at<myNum>(j,i) = -1;
                        }
                    }
                // Como las diferencias entre imágenes están en el rango de [-1,1]
                // Sumamos 1 a todos los valores para que el intervalo vaya de [0,2]
                residuals = residuals + 1.0f;

                // Aquí aplicaremos un mapeo proporcional a este intervalo
                // y le aplicamos una mascara de colores para observar las zonas
                // de mayor diferencia

                double min,max;
                cv::minMaxIdx(residuals, &min, &max);
                std::cout << "max: " << max << "min: " << min << std::endl;

                cv::Mat adjMap;
                cv::convertScaleAbs(residuals, adjMap, 255 / max);

                cv::Mat FalseColorMap;
                cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);

                // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
                // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

                cv::imshow("Diff", FalseColorMap);

                cv::waitKey();


                /** // Showing dual difference color maps
                cv::Mat FalseColorMap2;
                cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

                // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
                // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

                cv::imshow("Diff2", FalseColorMap2);
                **/
    } // Fin del bucle de las iteraciones

} // Fin de la funcion

// Calculo de la gradiente en dirección X(Horizontal)
// Se usa la fórmula de gradiente central
void Gradient(const cv::Mat & InputImg, cv::Mat & OutputXImg, cv::Mat & OutputYImg){
    // Creamos un Mat de Zeros con las mismas propiedades de InputImg
    int rows = InputImg.rows, cols = InputImg.cols;
    OutputXImg = cv::Mat::zeros(rows,cols,InputImg.type());
    OutputYImg = cv::Mat::zeros(rows,cols,InputImg.type());

    // Iteamos para calcular las gradientes
    // Observamos que no es posible calcular esta gradiente en los márgenes de la imagen
    for(int j = 1; j < rows-1; ++j)
        for(int i = 1; i < cols-1; ++i){
            // Gradiente en X
            OutputXImg.at<myNum>(j,i) = 0.5f * (InputImg.at<myNum>(j,i+1) - InputImg.at<myNum>(j,i-1));
            // Gradiente en Y
            OutputYImg.at<myNum>(j,i) = 0.5f * (InputImg.at<myNum>(j+1,i) - InputImg.at<myNum>(j-1,i));
        }

    // Visualizacion de estas imagenes
    /** // Codigo para visualizar las gradientes
    double min,max;
    OutputXImg = OutputXImg + 0.5f;
    cv::minMaxIdx(OutputXImg,&min,&max);
    //cout << "max:" << max << "min:" << min << endl;
    cv::Mat adjMap;
    OutputXImg.convertTo(adjMap,CV_8UC1,255/(max-min),-min); // Coloramiento de acuerdo a valores maximos y minimos

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);
    cv::cvtColor(FalseColorMap,FalseColorMap,CV_BGR2RGB);

    cv::imshow("Hola",FalseColorMap);

    OutputYImg = OutputYImg + 0.5f;
    cv::minMaxIdx(OutputYImg,&min,&max);
    cv::Mat adjMap2;
    OutputYImg.convertTo(adjMap2,CV_8UC1,255/(max-min),-min); // Coloramiento de acuerdo a valores maximos y minimos

    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap2,FalseColorMap2,cv::COLORMAP_BONE);
    cv::cvtColor(FalseColorMap2,FalseColorMap2,CV_BGR2RGB);

    cv::imshow("Hola2",FalseColorMap2);

    int key = cv::waitKey();
    **/
}

























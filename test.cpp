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


/*FUNCIONES*/
// Realiza el alineamiento de las imágenes
void doAlignment(const cv::Mat& i0ref, const cv::Mat& d0ref, const cv::Mat &i1ref, const cv::Mat &Kref ,const cv::Mat& i0, const cv::Mat& d0, const cv::Mat &i1, Eigen::VectorXd &xi, const cv::Mat &K);
// Retorna la imágen de residuales en base a un xi dado
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd &xi, const cv::Mat &K, Eigen::VectorXd& Res, Eigen::MatrixXd& Jac);
// funcion sobre cargadada para sólo visualizar
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd &xi, const cv::Mat &K);
// Estas funciones me ayudaran a calcular los gradientes en direccion X e Y para una imagen de entrada
void Gradient(const cv::Mat & InputImg, cv::Mat & OutputImg, cv::Mat &OutputYImg);
// funcion para realizar una interpolación lineal
void interpolate(const cv::Mat& InputImg, cv::Mat& OutputImg, const cv::Mat& map_x, const cv::Mat& map_y, int padding);

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
    //xi << -0.0018, 0.0065, 0.0369, -0.0287, -0.0184, -0.0004; // Resultado


    for(int lvl = 4; lvl >= 0; --lvl){
        std::cout << std::endl << "level = " << lvl << std::endl << std::endl;

        // proceso de downscale
        cv::Mat i0_scaled, i1_scaled, d0_scaled,K_scaled;
        i0_scaled = downscale(i0,lvl,intensity);
        i1_scaled = downscale(i1,lvl,intensity);
        d0_scaled = downscale(d0,lvl,depth);
        K_scaled = downscale(K,lvl,intrinsics);

        // Actualizamos el valor de xi
        doAlignment(i0,d0,i1,K,i0_scaled,d0_scaled,i1_scaled,xi,K_scaled);

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

// Alineamos una imagen a un par RGBD
void doAlignment(const cv::Mat& i0ref, const cv::Mat& d0ref, const cv::Mat &i1ref, const cv::Mat &Kref ,const cv::Mat& i0, const cv::Mat& d0, const cv::Mat &i1, Eigen::VectorXd &xi, const cv::Mat &K){

    double last_err = 10000000.0; // Vamos minimizando el error
    FOR(it,20){
        std::cout << "\niteracion " << it << std::endl << std::endl;
        Eigen::VectorXd R;
        Eigen::MatrixXd J;

        // Calculamos los residuales y el jacobiano
        // mostramos la imagen de los residuales
        CalcDiffImage(i0,d0,i1,xi,K,R,J);

        Eigen::MatrixXd J_inv = (J.transpose() * J);

        // Calculamos nuestro diferencial de xi
        Eigen::VectorXd d_xi = -J_inv.inverse() * J.transpose() * R;

        std::cout << "d_xi:\n" << d_xi.transpose() << std::endl;

        Eigen::VectorXd last_xi = xi;
        xi = rbm2twistcoord( twistcoord2rbm(d_xi) * twistcoord2rbm(xi));

        std::cout << "xi:\n" << xi.transpose() << std::endl;

        // Calculamos la media de todos los errores cuadráticos
        // de esta forma el error no estará ligado al numero de muestras
        // que varia en el cambio de cada nivel
        // ademas, solo actualizaremos si el error es menor

        double err = R.dot(R)/R.rows();
        std::cout << "err=" << err << " last_err=" << last_err << std::endl;

        CalcDiffImage(i0ref,d0ref,i1ref,xi,Kref);

        if( err / last_err > 0.995){
            //xi = last_xi;
            return;
        }

        last_err = err;

        cv::waitKey();

    }

} // Fin de la funcion doAlignment

// Funcion que calcula los residuales entre imágenes
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd &xi, const cv::Mat& K, Eigen::VectorXd &Res, Eigen::MatrixXd &Jac){
#ifdef enable_writting2file
    writeMat2File(i0,"trash_data/i0.txt");
    writeMat2File(d0,"trash_data/d0.txt");
    writeMat2File(i1,"trash_data/i1.txt");
    writeMat2File(K,"trash_data/K.txt");
#endif
    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;

    // Pasamos la matriz intrinseca a Eigen+
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<myNum>(0,0), K.at<myNum>(0,1), K.at<myNum>(0,2),
               K.at<myNum>(1,0), K.at<myNum>(1,1), K.at<myNum>(1,2),
               K.at<myNum>(2,0), K.at<myNum>(2,1), K.at<myNum>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();
    double fx = eigen_K(0,0), fy = eigen_K(1,1);
    //std::cout << "input xi=" << xi.transpose() << std::endl;

    // Calculamos la transformación rigid-body motion
    Eigen::Matrix4d g = twistcoord2rbm(xi);
    // Creamos nuestros mapeos para x e y
    cv::Mat map_warped_x, map_warped_y;
    map_warped_x.create(i1.size(), i1.type());
    map_warped_y.create(i1.size(), i1.type());

    // Y los mapeos para los Warp Coordinates(no proyectados)
    // restamos 100 para simular NaN values
    cv::Mat xp, yp, zp;
    xp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    yp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    zp = cv::Mat::zeros(i1.size(),i1.type()) - 100;

    // Calculamos las nuevas coordenadas
    FOR(j,rows){
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
                 } else {
                    map_warped_x.at<myNum>(j,i) = -100;
                    map_warped_y.at<myNum>(j,i) = -100;
                 } // Fin de Condicional exterior
        } // Fin Bucle FOR cols
    } // Fin Bucle FOR rows
#ifdef enable_writting2file
    writeMat2File(map_warped_x,"trash_data/warped_X_map.txt");
    writeMat2File(map_warped_y,"trash_data/warped_Y_map.txt");

    writeMat2File(xp,"trash_data/xp.txt");
    writeMat2File(yp,"trash_data/yp.txt");
    writeMat2File(zp,"trash_data/zp.txt");
#endif

    // Declaramos una matriz para guardar la interpolacion y despues los residuales
    cv::Mat i1_warped = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    // Realizamos la interpolación
    interpolate(i1,i1_warped,map_warped_x,map_warped_y,0);
    //cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));
#ifdef enable_writting2file
    writeMat2File(i1_warped,"trash_data/i1_warped.txt");
#endif


    cv::Mat residuals = cv::Mat::zeros(i1.size(),i1.type());
    residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!
#ifdef enable_writting2file
    writeMat2File(residuals,"trash_data/residuals.txt");
#endif


    // CALCULO DE LA GRADIENTE
    cv::Mat XGradient, YGradient;
    Gradient(i1,XGradient,YGradient);
#ifdef enable_writting2file
    writeMat2File(XGradient,"trash_data/XGradient.txt");
    writeMat2File(YGradient,"trash_data/YGradient.txt");
#endif

    // Interpolamos sobre las gradientes
    cv::Mat map_XGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    cv::Mat map_YGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    interpolate(XGradient,map_XGradient,map_warped_x,map_warped_y,1); // El padding es 1 por que no se pueden calcular
    interpolate(YGradient,map_YGradient,map_warped_x,map_warped_y,1); // valores de gradiente para los bordes

#ifdef enable_writting2file
    writeMat2File(map_XGradient,"trash_data/map_XGradient.txt");
    writeMat2File(map_YGradient,"trash_data/map_YGradient.txt");
#endif


    /*** CONSTRUCION DE c y r0 ***/
    // Formamos las matrices pero usando la libreria eigen
    Eigen::VectorXd r(rows * cols);
    Eigen::MatrixXd c(rows * cols,6);
    int cont = 0;
    FOR(i,cols){
        FOR(j,rows){
            // Sólo usaremos los datos que sean válidos
            // Es decir aquellos que tengan valores válidos de gradiente
            // 1 < map_warped_x < width -1; 1 < map_warped_y < height -1
            // Valores válidos para el Image warped
            // i1_warped != 0
            // Valores válidos para las coordenadas de pixel en 3D
            // xp,yp,zp != -100
            if( (1 < map_warped_x.at<myNum>(j,i) && map_warped_x.at<myNum>(j,i) < cols-2) &&
                (1 < map_warped_y.at<myNum>(j,i) && map_warped_y.at<myNum>(j,i) < rows-2) &&
                i1_warped.at<myNum>(j,i) != 0 &&
                xp.at<myNum>(j,i) != -100){
                    // Residuales
                    r(cont) = residuals.at<myNum>(j,i);

                    double gradX = map_XGradient.at<myNum>(j,i), gradY = map_YGradient.at<myNum>(j,i);
                    gradX *= fx; gradY *= fy;
                    double x = xp.at<myNum>(j,i), y = yp.at<myNum>(j,i), z = zp.at<myNum>(j,i);

                    // Jacobiano
                    c(cont,0) = gradX / z;
                    c(cont,1) = gradY / z;
                    c(cont,2) = -( gradX * x + gradY * y ) / (z*z);
                    c(cont,3) = -( gradX * x * y / (z*z)) -  (gradY * (1 + (y*y)/(z*z)));
                    c(cont,4) = ( gradX * (1 + (x*x)/(z*z))) + (gradY * x * y / (z*z));
                    c(cont,5) = (- gradX * y + gradY * x) / z;
            } else{
                r(cont) = 0.0;

                c(cont,0) = 0.0;
                c(cont,1) = 0.0;
                c(cont,2) = 0.0;
                c(cont,3) = 0.0;
                c(cont,4) = 0.0;
                c(cont,5) = 0.0;
            }
            cont++;
        } // Fin bucle for Cols
    }// Fin bucle for Rows

    //std::cout << "cont: <<" << cont << std::endl;

    /** Retornamos los Residuales y el Jacobiano **/
    // Hacemos un slice con el conteo de "cont" pixeles validos
    Res = r;
    Jac = c; Jac = -Jac;
#ifdef enable_writting2file
    writeEigenVec2File(Res,"trash_data/Res.txt");
    writeEigenMat2File(Jac,"trash_data/Jac.txt");
#endif

    /* displaying ressults */

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
    //std::cout << "max: " << max << "min: " << min << std::endl;

    cv::Mat adjMap;
    cv::convertScaleAbs(residuals, adjMap, 255 / max);

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff", FalseColorMap);

    //cv::waitKey();

    /** // Showing dual difference color maps
    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff2", FalseColorMap2);
    **/

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


void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd &xi, const cv::Mat &K){
    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;

    // Pasamos la matriz intrinseca a Eigen+
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<myNum>(0,0), K.at<myNum>(0,1), K.at<myNum>(0,2),
               K.at<myNum>(1,0), K.at<myNum>(1,1), K.at<myNum>(1,2),
               K.at<myNum>(2,0), K.at<myNum>(2,1), K.at<myNum>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();

    // Declaramos una matriz para guardar los residuales
    cv::Mat i1_warped = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    // Calculamos la transformación rigid-body motion
    Eigen::Matrix4d g = twistcoord2rbm(xi);
    // Creamos nuestros mapeos para x e y
    cv::Mat map_warped_x, map_warped_y;
    map_warped_x.create(i1.size(), i1.type());
    map_warped_y.create(i1.size(), i1.type());

    // Y los mapeos para los Warp Coordinates(no proyectados)
    // restamos 100 para simular NaN values
    cv::Mat xp, yp, zp;
    xp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    yp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    zp = cv::Mat::zeros(i1.size(),i1.type()) - 100;

    // Calculamos las nuevas coordenadas
    FOR(j,rows){
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

                 } else {
                    map_warped_x.at<myNum>(j,i) = -100;
                    map_warped_y.at<myNum>(j,i) = -100;
                 } // Fin de Condicional exterior
        } // Fin Bucle FOR cols
    } // Fin Bucle FOR rows

    // Interpolamos los valores para los warped coordinates
    // dejaremos la interpolacion de opencv porque aqui solo la usamos para visualizar
    cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat residuals = cv::Mat::zeros(i1.size(),i1.type());
    residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!

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

    cv::imshow("Actual comparison", FalseColorMap);

    //cv::waitKey();

    /** // Showing dual difference color maps
    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff2", FalseColorMap2);
    **/

}



void interpolate(const cv::Mat& InputImg, cv::Mat& OutputImg, const cv::Mat& map_x, const cv::Mat& map_y, int padding){
    double warp_coord_x, warp_coord_y;
    FOR(j,InputImg.rows){
        FOR(i,InputImg.cols){
            // Debemos corregir los valores de mynum
            warp_coord_x = map_x.at<myNum>(j,i);
            warp_coord_y = map_y.at<myNum>(j,i);
            // Revisamos si las coord se encuentran dentro de los limites de la imagen
            // Considerando el padding
            if( warp_coord_x > padding && warp_coord_x < InputImg.cols - 1 - padding &&
                warp_coord_y > padding && warp_coord_y < InputImg.rows - 1 - padding ){
                double a = warp_coord_y - floor(warp_coord_y);
                double b = warp_coord_x - floor(warp_coord_x);
                int t = floor(warp_coord_y), d = ceil(warp_coord_y);
                int s = floor(warp_coord_x), r = ceil(warp_coord_x);

                double t_s = InputImg.at<myNum>(t,s);
                double d_s = InputImg.at<myNum>(d,s);
                double t_r = InputImg.at<myNum>(t,r);
                double d_r = InputImg.at<myNum>(d,r);

                double result = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);

                //std::cout << result << " ";

                OutputImg.at<myNum>(j,i) = result;

            } else {
                OutputImg.at<myNum>(j,i) = 0.0;
            }

        } // Fin del FOR interior
    } // Fin del FOR exterior

}




















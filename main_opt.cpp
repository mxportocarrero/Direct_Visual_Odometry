/**
  Direct Odometry from RGB-D Images
  Autor: Max W. Portocarrero
  Compilamos con el siguiente comando
  // Automatic Optimization
  g++ main_opt.cpp dataset.cpp -o main_opt `pkg-config opencv --cflags --libs` -O3 -funsafe-math-optimizations -mavx2
  execute with
  ./main_opt.cpp > main_opt.out
    avg fps = 12
    max fps = 14-15
  // Run Executable Command
  ./main_opt > main_opt.out
**/

// Import Standart Libraries
#include <iostream>

#include "utilities.hpp"
#include "linear_algebra_functions.hpp"
#include "dataset.hpp"

// Habilitar la opcion para escribir las diferentes matrices en archivos
#define enable_writting2file

//#define DATABASE_NAME "data/burghers_sample_png"
//#define DATABASE_NAME "data/cactusgarden_png"
#define DATABASE_NAME "data/rgbd_dataset_freiburg11_room"
//#define DATABASE_NAME "data/rgbd_dataset_freiburg1_xyz"
//#define DATABASE_NAME "data/rgbd_dataset_freiburg2_desk"
//#define DATABASE_NAME "test_data/rgbd_dataset_freiburg2_desk_output"

enum image_type {intensity, depth, intrinsics};
cv::Mat downscale(const cv::Mat & image, int level, int type);
// Este downscale solo tomara una imagen y la reducira en la mitad
cv::Mat downscale2(const cv::Mat& img,int type);


/*FUNCIONES*/
// Realiza el alineamiento de las imágenes
void doAlignment(const cv::Mat& i0ref, const cv::Mat& d0ref, const cv::Mat &i1ref, const cv::Mat &Kref ,const cv::Mat& i0, const cv::Mat& d0, const cv::Mat &i1, Eigen::VectorXd &xi, const cv::Mat &K);
// Retorna la imágen de residuales en base a un xi dado
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const cv::Mat &XGradient, const cv::Mat &YGradient, const Eigen::VectorXd &xi, const cv::Mat &K, Eigen::VectorXd& Res, Eigen::MatrixXd& Jac);
// funcion sobre cargadada para sólo visualizar
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const Eigen::VectorXd &xi, const cv::Mat &K);
// Estas funciones me ayudaran a calcular los gradientes en direccion X e Y para una imagen de entrada
void Gradient(const cv::Mat & InputImg, cv::Mat & OutputImg, cv::Mat &OutputYImg);
// funcion para realizar una interpolación lineal
void interpolate(const cv::Mat& InputImg, cv::Mat& OutputImg, const cv::Mat& map_x, const cv::Mat& map_y, int padding);

int main(){

    // Usamos la clase Dataset para leer los pares RGBD
    Dataset myDataset(DATABASE_NAME);

    // Parámetros intrinsecos para estas imágenes
    float k[3][3] = {{517.3f,   0.0f, 318.6f},
                     {  0.0f, 516.5f, 255.3f},
                     {  0.0f,   0.0f,   1.0f}};
    float k_fr2[3][3] = {{520.9f,   0.0f, 325.1f},
                     {  0.0f, 521.0f, 249.7f},
                     {  0.0f,   0.0f,   1.0f}};

    //Expresados en cv::Mat
    cv::Mat K = cv::Mat(cv::Size(3,3),CV_32F,k_fr2); // Ojo que matriz estemos usando!

    // Vamos a escribir los xi en un archivo separado
    std::ofstream fout;

    // Declaramos algunas variables generales
    cv::Mat i0,i1;
    cv::Mat d0;
    Eigen::VectorXd xi(6);
    xi << 0,0,0,0,0,0;

    fout.open("odometry.txt",std::fstream::app);
    fout.precision(4);
    if(fout.is_open()){
        fout << myDataset.getTimestamp_filename(0) << " ";
        fout << xi.transpose() << "\n";
    }
    fout.close();

    auto start = cv::getTickCount();

    for(int frame=1; frame < myDataset.NoFrames(); ++frame){
        std::cout << "\n***  FRAME " << frame << "   ***\n";
        // usando opencv
        // El algoritmo que usa para pasar a grayscale es distinto al de Matlab
        if(read_image(i0,myDataset.getRGB_filename(frame-1))){
           //cv::imshow("Display Image i0",i0); // activar para ver ambos pares
        }
        if(read_image(i1,myDataset.getRGB_filename(frame))){
            cv::imshow("Display Image i1",i1);
        }

        if(read_depth_image(d0,myDataset.getDEPTH_filename(frame-1),5000.0f)){
            show_depth_image("depth at t0",d0);
        }

        xi << 0,0,0,0,0,0; // seting xi to initial vales¿¿ues
        //xi << 1,2,3,4,5,6;
        //xi << -0.0018, 0.0065, 0.0369, -0.0287, -0.0184, -0.0004; // Resultado

        // Obtaining Initial Pyramidal Images
        std::vector<cv::Mat> vo_img_ref,vo_img,vo_depth,vo_int;

        auto s = cv::getTickCount();
        vo_img_ref.push_back(downscale2(i0,intensity));
        vo_img.push_back(downscale2(i1,intensity));
        vo_depth.push_back(downscale2(d0,depth));
        vo_int.push_back(downscale2(K,intrinsics));

        FOR(i,3){
            vo_img_ref.push_back(downscale2(vo_img_ref.back(),intensity));
            vo_img.push_back(downscale2(vo_img.back(),intensity));
            vo_depth.push_back(downscale2(vo_depth.back(),depth));
            vo_int.push_back(downscale2(vo_int.back(),intrinsics));
        }
        auto e = cv::getTickCount();
        double t = (e - s) / cv::getTickFrequency();
        std::cout << "Downscale Process Time: " << t / 10.0 << " seconds\n";

        // Obtaining Alignment - Updating xi
        for(int lvl = 4; lvl >= 0; --lvl){
            std::cout << std::endl << "level = " << lvl << std::endl << std::endl;

            if(lvl > 0)
                doAlignment(i0,d0,i1,K,vo_img_ref[lvl-1],vo_depth[lvl-1],vo_img[lvl-1],xi,vo_int[lvl-1]);
            else
                doAlignment(i0,d0,i1,K,i0,d0,i1,xi,K);

        } // Fin de Bucle de niveles


        // Vamos agregando los xi obtenidos en cada iteración
        fout.open("odometry.txt",std::fstream::app);
        fout.precision(4);
        if(fout.is_open()){
            fout << myDataset.getTimestamp_filename(frame) << " ";
            //fout << xi.transpose() << "\n";
            fout << xi(0) << " " << xi(1) << " " << xi(2) << " " << xi(3) << " " << xi(4) << " " << xi(5) << "\n";
        }
        fout.close();


        cv::waitKey(1);

        if(frame % 10 == 0){
            auto end = cv::getTickCount();
            double time = (end - start) / cv::getTickFrequency();
            std::cout << "Mean Process Time per Frame: " << time / 10.0 << " seconds\n";
            double fps = (1.0 / time) * 10.0;
            std::cout << "FPS: " << fps << std::endl;
            start = end; // Reiniciamos el conteo
        }

    } // Fin de Bucle de Frames


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

cv::Mat downscale2(const cv::Mat & image, int type){
    switch(type){
        case intensity:{
            int rows = image.rows;
            int cols = image.cols;

            // creamos una matriz que calcula la mitad de la imagen
            cv::Mat scaled_image = cv::Mat::zeros(cv::Size(cols/2,rows/2),image.type());

            double num;
            for(int j = 0; j < rows/2; j++){
                for(int i = 0; i < cols/2; i++){
                    num = image.at<myNum>(2*j,2*i) +
                                 image.at<myNum>(2*j+1,2*i) +
                                 image.at<myNum>(2*j,2*i+1) +
                                 image.at<myNum>(2*j+1,2*i+1);
                    scaled_image.at<myNum>(j,i) = num / 4;
                 }
             }

            return scaled_image;
            break;
        }
        case depth:{

            int rows = image.rows;
            int cols = image.cols;

            // creamos una matriz que calcula la mitad de la imagen
            cv::Mat scaled_image = cv::Mat::zeros(cv::Size(cols/2,rows/2),image.type());

            double num;
            myNum a,b,c,d;
            int cont;
            for(int j = 0; j < rows/2; j++){
                for(int i = 0; i < cols/2; i++){
                    cont = 0;
                    //Contamos la cantidad de pixeles no nulos en la vecindad del pixel
                    a = image.at<myNum>(2*j,2*i);
                    b = image.at<myNum>(2*j+1,2*i);
                    c = image.at<myNum>(2*j,2*i+1);
                    d = image.at<myNum>(2*j+1,2*i+1);
                    if(a)
                        cont++;
                    if(b)
                        cont++;
                    if(c)
                        cont++;
                    if(d)
                        cont++;

                    num = a + b + c + d;

                    if(cont == 0.0f) scaled_image.at<myNum>(j,i) = 0.0f;
                    else scaled_image.at<myNum>(j,i) = num / cont;
                }
            }
            return scaled_image;
            break;
        }
        // En el caso de la matriz intrinseca es una matrix 3 x 3
        case intrinsics:{
            cv::Mat scaled_intrinsics = cv::Mat::eye(cv::Size(3,3),image.type());

            scaled_intrinsics.at<float>(0,0) = image.at<float>(0,0) / 2.0f;
            scaled_intrinsics.at<float>(1,1) = image.at<float>(1,1) / 2.0f;
            scaled_intrinsics.at<float>(0,2) = (image.at<float>(0,2) + 0.5f) / 2.0f - 0.5f;
            scaled_intrinsics.at<float>(1,2) = (image.at<float>(1,2) + 0.5f) / 2.0f - 0.5f;

            return scaled_intrinsics;
            break;
        }
        default:{
            break;
        }
    }

} // Fin de Dowscale2

// Alineamos una imagen a un par RGBD
void doAlignment(const cv::Mat& i0ref, const cv::Mat& d0ref, const cv::Mat &i1ref, const cv::Mat &Kref ,const cv::Mat& i0, const cv::Mat& d0, const cv::Mat &i1, Eigen::VectorXd &xi, const cv::Mat &K){
    // Calculamos la gradiente in-advance
    // CALCULO DE LA GRADIENTE
    cv::Mat XGradient, YGradient;
    Gradient(i1,XGradient,YGradient);

    double last_err = 10000000.0; // Vamos minimizando el error
    FOR(it,20){
        std::cout << "\niteracion " << it << std::endl << std::endl;
        Eigen::VectorXd R;
        Eigen::MatrixXd J;

        // Calculamos los residuales y el jacobiano
        // mostramos la imagen de los residuales
        auto start = cv::getTickCount();
        CalcDiffImage(i0,d0,i1,XGradient,YGradient,xi,K,R,J);
        auto end = cv::getTickCount();
        double time = (end - start) / cv::getTickFrequency();
        std::cout << "Residual and Jacobian Process Time: " << time << " seconds\n";

        // Calculamos nuestro diferencial de xi
        start = cv::getTickCount();
        Eigen::VectorXd d_xi = -(J.transpose() * J).inverse() * J.transpose() * R;
        end = cv::getTickCount();
        time = (end - start) / cv::getTickFrequency();
        std::cout << "d_xi Process Time: " << time << " seconds\n";

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

        // Visualizacion de los residuales
        //CalcDiffImage(i0ref,d0ref,i1ref,xi,Kref);

        if( err / last_err > 0.995){
            //xi = last_xi;
            return;
        }

        last_err = err;

        cv::waitKey(); // activar aqui para realizar una inspeccion frame a frame

    }

} // Fin de la funcion doAlignment

// Funcion que calcula los residuales entre imágenes
void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1,const cv::Mat& XGradient, const cv::Mat& YGradient, const Eigen::VectorXd &xi, const cv::Mat& K, Eigen::VectorXd &Res, Eigen::MatrixXd &Jac){

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
    cv::Mat map_warped_x(i1.size(),i1.type(),-100.0);
    cv::Mat map_warped_y(i1.size(),i1.type(),-100.0);

    // Y los mapeos para los Warp Coordinates(no proyectados)
    // restamos 100 para simular NaN values
    cv::Mat xp(i1.size(),i1.type(),-100.0);
    cv::Mat yp(i1.size(),i1.type(),-100.0);
    cv::Mat zp(i1.size(),i1.type(),-100.0);

    auto start = cv::getTickCount();
    //Eigen::Vector2d coord0;
    Eigen::Vector3d world_coord;
    Eigen::Vector4d transformed_coord;
    Eigen::Vector3d projected_coord;
    //Eigen::Vector2d warped_coord;
    // Calculamos las nuevas coordenadas
    double de;
    double* w_c0 = &world_coord(0);
    double* w_c1 = &world_coord(1);
    double* w_c2 = &world_coord(2);
    double* t_c0 = &transformed_coord(0);
    double* t_c1 = &transformed_coord(1);
    double* t_c2 = &transformed_coord(2);
    transformed_coord(3) = 1;
    double* p_c0 = &projected_coord(0);
    double* p_c1 = &projected_coord(1);
    double* p_c2 = &projected_coord(2);
    FOR(j,rows){
        const myNum* d = d0.ptr<myNum>(j);
        myNum* map_w_x = map_warped_x.ptr<myNum>(j);
        myNum* map_w_y = map_warped_y.ptr<myNum>(j);

        myNum* x = xp.ptr<myNum>(j);
        myNum* y = yp.ptr<myNum>(j);
        myNum* z = zp.ptr<myNum>(j);
        FOR(i,cols){
            de = d[i]; // guardamos temporalmente este valor para no estarlo calculando a cada momento
            if( d[i] > 0 ){
                // El problema con usar las asignaciones del Eigen
                // es q no son eficientes. lo que podemos salvarlo al usar punteros
                // el codigo se hace un poco engorroso pero es mas liviano
                // dejamos que eigen se haga cargo sólo de las multiplicaciones

                 //coord0 << i, j;
                 //std::cout << "x,y " << coord0 << " ;";

                 //world_coord << d[i] * i, d[i] * j, d[i];
                 *w_c0 = de * i; *w_c1 = de * j; *w_c2 = de;
                 //world_coord = d[i] * world_coord;
                 world_coord = eigen_K_inverse * world_coord;
                 //std::cout << world_coord << " ;";

                 // Transformed coord by rigid-body motion
                 //transformed_coord << world_coord, 1;
                 *t_c0 = *w_c0; *t_c1 = *w_c1; *t_c2 = *w_c2;
                 transformed_coord = g * transformed_coord;
                 //std::cout << transformed_coord << " ;";

                 //projected_coord << *t_c0,*t_c1,*t_c2;
                 *p_c0=*t_c0; *p_c1=*t_c1; *p_c2=*t_c2;
                 projected_coord = eigen_K * projected_coord;
                 //std::cout << projected_coord << " ;";

                 //warped_coord << projected_coord(0) / projected_coord(2), projected_coord(1) / projected_coord(2);
                 //std::cout << warped_coord << " ;\n";

                 // Probemos usar los mapeos de opencv
                 map_w_x[i] = *p_c0 / *p_c2;
                 map_w_y[i] = *p_c1 / *p_c2;

                 // Verificamos que el número sea positivo
                 // Para que no haya problema al calcular el Jacobiano
                 //if(t_c2 > 0.0f){
                     x[i] = *t_c0;
                     y[i] = *t_c1;
                     z[i] = *t_c2;
                 //}
            }
        } // Fin Bucle FOR cols
    } // Fin Bucle FOR rows
    auto end = cv::getTickCount();
    double time = (end - start) / cv::getTickFrequency();
    std::cout << "Warping Process Time: " << time << "seconds" << std::endl;
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
    start = cv::getTickCount();
    interpolate(i1,i1_warped,map_warped_x,map_warped_y,0);
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    std::cout << "1st Interpolation Process Time: " << time << "seconds" << std::endl;
    //cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));
#ifdef enable_writting2file
    writeMat2File(i1_warped,"trash_data/i1_warped.txt");
#endif


    cv::Mat residuals = cv::Mat::zeros(i1.size(),i1.type());
    residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!
#ifdef enable_writting2file
    writeMat2File(residuals,"trash_data/residuals.txt");
#endif


#ifdef enable_writting2file
    writeMat2File(XGradient,"trash_data/XGradient.txt");
    writeMat2File(YGradient,"trash_data/YGradient.txt");
#endif

    // Interpolamos sobre las gradientes
    cv::Mat map_XGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    cv::Mat map_YGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    start = cv::getTickCount();
    interpolate(XGradient,map_XGradient,map_warped_x,map_warped_y,1); // El padding es 1 por que no se pueden calcular
    interpolate(YGradient,map_YGradient,map_warped_x,map_warped_y,1); // valores de gradiente para los bordes
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    std::cout << "2nd BiInterpolation Process Time: " << time << "seconds" << std::endl;

#ifdef enable_writting2file
    writeMat2File(map_XGradient,"trash_data/map_XGradient.txt");
    writeMat2File(map_YGradient,"trash_data/map_YGradient.txt");
#endif

    start = cv::getTickCount();

    /*** CONSTRUCION DE c y r0 ***/
    // Formamos las matrices pero usando la libreria eigen
    Eigen::VectorXd r(rows * cols);
    Eigen::MatrixXd c(rows * cols,6);
    int cont = 0;
    double map_w_x, map_w_y;
    double gradX, gradY;
    double x,y,z;
    FOR(j,rows){
        myNum* res = residuals.ptr<myNum>(j);
        myNum* m_w_x = map_warped_x.ptr<myNum>(j);
        myNum* m_w_y = map_warped_y.ptr<myNum>(j);
        myNum* x_ = xp.ptr<myNum>(j);
        myNum* y_ = yp.ptr<myNum>(j);
        myNum* z_ = zp.ptr<myNum>(j);
        myNum* xgrad = map_XGradient.ptr<myNum>(j);
        myNum* ygrad = map_YGradient.ptr<myNum>(j);
        FOR(i,cols){
            // Sólo usaremos los datos que sean válidos
            // Es decir aquellos que tengan valores válidos de gradiente
            // 1 < map_warped_x < width -1; 1 < map_warped_y < height -1
            // Valores válidos para el Image warped
            // i1_warped != 0
            // Valores válidos para las coordenadas de pixel en 3D
            // xp,yp,zp != -100
            map_w_x = m_w_x[i];
            map_w_y = m_w_y[i];

            if( (1 < map_w_x && map_w_x < cols-2) &&
                (1 < map_w_y && map_w_y < rows-2) &&
                i1_warped.at<myNum>(j,i) != 0 &&
                x_[i] != -100){
                    // Residuales
                    r(cont) = res[i];

                    gradX = xgrad[i]; gradY = ygrad[i];
                    gradX *= fx; gradY *= fy;
                    x = x_[i]; y = y_[i]; z = z_[i];

                    // Jacobiano
                    c(cont,0) = gradX / z;
                    c(cont,1) = gradY / z;
                    c(cont,2) = -( gradX * x + gradY * y ) / (z*z);
                    c(cont,3) = -( gradX * x * y / (z*z)) -  (gradY * (1 + (y*y)/(z*z)));
                    c(cont,4) = ( gradX * (1 + (x*x)/(z*z))) + (gradY * x * y / (z*z));
                    c(cont,5) = (- gradX * y + gradY * x) / z;

                    cont++;
            }
        } // Fin bucle for Cols
    }// Fin bucle for Rows
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    std::cout << "R0 y C Interpolation Process Time: " << time << "seconds" << std::endl;

    /** Retornamos los Residuales y el Jacobiano **/
    // Hacemos un slice con el conteo de "cont" pixeles validos
    Res = r.block(0,0,cont,1);
    Jac = c.block(0,0,cont,6); Jac = -Jac;
#ifdef enable_writting2file
    writeEigenVec2File(Res,"trash_data/Res.txt");
    writeEigenMat2File(Jac,"trash_data/Jac.txt");
#endif

    /**
    // displaying ressults

    // FILTRAMOS LOS VALORES CON VALOR 0 DESPUES DE LA INTERPOLACION
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
    **/ // FIN DE VISUALIZACIÓN BÁSICA

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
    for(int j = 1; j < rows-1; ++j){
        myNum* grad_x = OutputXImg.ptr<myNum>(j);
        myNum* grad_y = OutputYImg.ptr<myNum>(j);

        const myNum* input_j = InputImg.ptr<myNum>(j);
        const myNum* input_ja = InputImg.ptr<myNum>(j+1);
        const myNum* input_jb = InputImg.ptr<myNum>(j-1);

        for(int i = 1; i < cols-1; ++i){
            // Gradiente en X
            grad_x[i] = 0.5f * (input_j[i+1] - input_j[i-1]);
            // Gradiente en Y
            grad_y[i] = 0.5f * (input_ja[i] - input_jb[i]);
        }
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
    //double result;
    double a, b, t, s, d, r;
    double t_s, d_s, t_r, d_r;
    FOR(j,InputImg.rows){
        const myNum* pixel_x = map_x.ptr<myNum>(j);
        const myNum* pixel_y = map_y.ptr<myNum>(j);
        myNum* pixel_out = OutputImg.ptr<myNum>(j);
        FOR(i,InputImg.cols){
            // Debemos corregir los valores de mynum
            warp_coord_x = pixel_x[i];
            warp_coord_y = pixel_y[i];
            // Revisamos si las coord se encuentran dentro de los limites de la imagen
            // Considerando el padding
            if( warp_coord_x > padding && warp_coord_x < InputImg.cols - 1 - padding &&
                warp_coord_y > padding && warp_coord_y < InputImg.rows - 1 - padding ){
                a = warp_coord_y - floor(warp_coord_y);
                b = warp_coord_x - floor(warp_coord_x);
                t = floor(warp_coord_y), d = ceil(warp_coord_y);
                s = floor(warp_coord_x), r = ceil(warp_coord_x);

                t_s = InputImg.at<myNum>(t,s);
                d_s = InputImg.at<myNum>(d,s);
                t_r = InputImg.at<myNum>(t,r);
                d_r = InputImg.at<myNum>(d,r);

                //result = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);

                //*(pixel_out+i) = result;
                //pixel_out[i] = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);

                // Podemos ahorrarnos unas cuantas instrucciones si efectuamos algo de algebra para simplificar
                // el calculo de la interpolacion, reduciendo el numero de multiplicaciones, adiciones y asignaciones

                pixel_out[i] = ( t_r + a*(d_r-t_r) )*b + ( t_s + a*(d_s-t_s) )*(1-b);
            }
        } // Fin del FOR interior
    } // Fin del FOR exterior

}

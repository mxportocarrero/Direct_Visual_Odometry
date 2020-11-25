# Direct_Visual_Odometry
Implementacion del paper de Steinbrucker 2011


Compilamos con el siguiente comando

    g++ main.cpp dataset.cpp -o main `pkg-config opencv --cflags --libs` 

y ejecutamos con

    ./main

como la secuencias esta hardcodeada en las definiciones cada vez que se quiera probar otra secuencia hay que recompilar

## Issues

como el ejecutable se compila con make file, se debe generar un archivo opecv.pc el cual contiene las configurariones y paths de las librerias isntaladas por opencv. para ello se debe compilar el opencv con el siguiente flag

    -D OPENCV_GENERATE_PKGCONFIG=ON

esto crea el archivo opencv4.pc (para la version 4) en la ruta /(install-dir)/lib/pkgconfig/opencv4.pc

para regristrarlo debemos exportar ese path a nuestras varibles de entorno

    export PKG_CONFIG_PATH=~/OpenCV-4.0/lib/pkgconfig/

registramos el pkg

    pkg-config --cflags --libs opencv4

antes de compilar observar que el pkg-config apunte a opencv4, depende de la version

    g++ main.cpp dataset.cpp -o main `pkg-config opencv4 --cflags --libs`

Durante la compilación me surgio otro error, al tratar de usar Eigen. el cual se instalo con

    sudo apt-get install libeigen3-dev

    In file included from general_includes.hpp:26:0,
                 from utilities.hpp:10,
                 from main.cpp:11:
    /usr/include/eigen3/unsupported/Eigen/MatrixFunctions:17:10: fatal error: Eigen/Core: No such file or directory
    #include <Eigen/Core>

Este error era por un symlink que no estaba haciendo buena referencia por lo que solo acepta el siguiente include

    #include <eigen3/Eigen/core>

se podia solucionar cambiando en los headers de la libreria en el lugar de instalacion. pero decidi crear un symlink adecuado para no modificar ello

    sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

Seguidamente, surgio un error de sintaxis, fue que las constantes para opencv4 fueron actualizadas por lo que cambie 

    cv::IMREAD_COLOR -> CV_LOAD_IMAGE_COLOR
    cv::IMREAD_ANYDEPTH -> CV_LOAD_IMAGE_ANYDEPTH

Por ultimo, el SO no podia encontrar las shared libraries necesarias para producir el binario adecuado, asi que tuve que crear un archivo .conf
y recargar los paths para los shared libraries (.so)

    ./main: error while loading shared libraries: libopencv_highgui.so.4.4: cannot open shared object file: No such file or directory

    cd /etc/ld.so.conf.d/
    sudo nano opencv.conf

    ## Agregar el siguiente path a este archivo
    ## Apuntando al lugar donde se encuentran los archivos .so
    /home/maxideveloper/OpenCV-4.0/lib/

    ## y para recargar los ld.so.conf
    sudo ldconfig -v

Con eso ya se deberia poder observar que se esten haciendo referencia a los archivos .so necesarios



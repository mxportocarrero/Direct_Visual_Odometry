#ifndef LINEAR_ALGEBRA_FUNCTIONS_HPP
#define LINEAR_ALGEBRA_FUNCTIONS_HPP

//#include "linear_algebra_functions.hpp"

// Elegimos si usamos la operación exponencial o las formulas
bool using_exp_mapping = false;

/*  // Incluir estas lineas para testear las funciones de transformacion
    Eigen::Vector3d w(4,5,6);
    Eigen::Matrix3d R = vector2rotation(w);
    w = rotation2vector(R);
    vector2rotation(w);

    Eigen::VectorXd xi(6);
    xi << 1,2,3,4,5,6;
    Eigen::Matrix4d g = twistcoord2rbm(xi);
    xi = rbm2twistcoord(g);
    twistcoord2rbm(xi);
*/

Eigen::Matrix3d hat(const Eigen::Vector3d & w){
    Eigen::Matrix3d w_hat;
    w_hat <<     0, -w(2),  w(1),
              w(2),     0, -w(0),
             -w(1),  w(0),     0;
    //std::cout << "mat = " << std::endl << w_hat << std::endl;
    return w_hat;
}


// Transformaciones de Rotacion
Eigen::Matrix3d vector2rotation(const Eigen::Vector3d & w){
    Eigen::Matrix3d R;
    Eigen::Matrix3d w_hat = hat(w);

    if(using_exp_mapping){
        // usando la matrix exponencial de Eigen
        R = w_hat.exp();
    }
    else{
        myNum w_norm = w.norm();

        if(w_norm == 0)
            R = Eigen::Matrix3d::Identity();
        else{
            //R = Eigen::Matrix3d::Zero(); // forma para crear matrices de ceros en eigen
            R = Eigen::Matrix3d::Identity();
            R += w_hat / w_norm * sin(w_norm);
            R += w_hat.pow(2) / pow(w_norm,2) * (1 - cos(w_norm));
        }

    }
    //std::cout << "R = " << std::endl << R << std::endl;
    return R;
}

Eigen::Vector3d rotation2vector(const Eigen::Matrix3d & R){
    Eigen::Vector3d w;

    if(using_exp_mapping){
        Eigen::Matrix3d w_hat = R.log();
        //std::cout << "w_hat = " << std::endl << w_hat << std::endl;
        w << -w_hat(1,2), w_hat(0,2), -w_hat(0,1);
    }else{
        myNum w_norm = acos((R.trace() - 1) / 2);
        if(w_norm == 0){
            w = Eigen::Vector3d(0,0,0);

        }else{
            Eigen::Vector3d A(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
            w = w_norm * (1 / (2 * sin(w_norm))) * A;
        }
    }
    //std::cout << "w = " << std::endl << w << std::endl;
    return w;
}

// Transformaciones de Rigid-Body Motions

Eigen::Matrix4d twistcoord2rbm(const Eigen::VectorXd & xi){
    Eigen::Matrix4d g;
    if(using_exp_mapping){
            Eigen::Matrix4d xi_hat;
            xi_hat <<   0    , -xi(5) ,  xi(4)  , xi(0),
                       xi(5) ,   0    , -xi(3)  , xi(1),
                      -xi(4) ,  xi(3) ,   0     , xi(2),
                        0    ,   0    ,   0     , 0;

            g = xi_hat.exp();
    }else{
        // formamos el vector v y w
        Eigen::Vector3d v; v << xi(0),xi(1),xi(2);
        Eigen::Vector3d w; w << xi(3),xi(4),xi(5);

        Eigen::Matrix3d R;
        Eigen::Vector3d T;

        myNum w_norm = w.norm();

        if(w_norm == 0){
            R = Eigen::Matrix3d::Identity();
            T = v;
        }else{
            R = vector2rotation(w);
            T = ((Eigen::Matrix3d::Identity() - R) * hat(w) + w * w.transpose()) * v / pow(w_norm,2);
        }
        g << R(0,0) , R(0,1) , R(0,2) , T(0),
             R(1,0) , R(1,1) , R(1,2) , T(1),
             R(2,0) , R(2,1) , R(2,2) , T(2),
               0    ,   0    ,   0    , 1;
    }
    //std::cout << "g = " << std::endl << g << std::endl;
    return g;
}

Eigen::VectorXd rbm2twistcoord(const Eigen::Matrix4d & g){
    Eigen::VectorXd xi(6);
    Eigen::Vector3d v, w;
    if(using_exp_mapping){
        Eigen::Matrix4d xi_hat = g.log();
        w = Eigen::Vector3d(-xi_hat(1,2),xi_hat(0,2),-xi_hat(0,1));
        v = Eigen::Vector3d(xi_hat(0,3),xi_hat(1,3),xi_hat(2,3));
    }else{
        Eigen::Matrix3d R;
        Eigen::Vector3d T;

        R << g(0,0), g(0,1), g(0,2),
             g(1,0), g(1,1), g(1,2),
             g(2,0), g(2,1), g(2,2);

        T << g(0,3), g(1,3), g(2,3);

        w = rotation2vector(R);

        myNum w_norm = w.norm();

        if(w_norm == 0)
            v = T;
        else{
            Eigen::Matrix3d A = (Eigen::Matrix3d::Identity() - R) * hat(w) + w * w.transpose();
            v = pow(w_norm,2) * A.inverse() * T;
        }
    }

    xi << v, w;

    //std::cout << "xi = " << std::endl << xi << std::endl;

    return xi;
}

#endif // LINEAR_ALGEBRA_FUNCTIONS_HPP

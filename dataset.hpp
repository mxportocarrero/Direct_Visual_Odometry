#ifndef DATASET_HPP
#define DATASET_HPP

// Esta clase se encarga de leer el archivo que contiene
// los archivos de los pares RGB-D sincronizados

#include "general_includes.hpp"


class Dataset
{
private:
    std::string database_name;

    std::vector<std::string> rgb_filenames;
    std::vector<std::string> depth_filenames;

public:
    Dataset(std::string folder_name);

    std::string getRGB_filename(int i);
    std::string getDEPTH_filename(int i);

    int NoFrames();

};

#endif // DATASET_HPP

#include "dataset.hpp"


std::vector<std::string> split(const std::string& s, char separator)
{
   std::vector<std::string> output;

    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(separator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );

        output.push_back(substring);

        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word

    return output;
}

Dataset::Dataset(std::string folder_name)
{
    database_name = folder_name;

    std::ifstream myRGBfile;
    std::ifstream myDEPTHfile;

    myRGBfile.open(folder_name + "/rgb_t.txt");
    myDEPTHfile.open(folder_name + "/depth_t.txt");

    std::string RGBline;
    std::string DEPTHline;
    if(myRGBfile.is_open()){
        while (std::getline(myRGBfile,RGBline) && std::getline(myDEPTHfile,DEPTHline)) {
            //std::cout << RGBline +" "+ DEPTHline << "\n";
            rgb_filenames.push_back( split(RGBline,' ')[1] );
            timestamp_filenames.push_back(split(RGBline,' ')[0]); // vamos a usar los timestamps de las imagenes, por los depth están registradas a las imagenes
            depth_filenames.push_back( split(DEPTHline,' ')[1]);
        }
        //Eliminamos las cabeceras
        rgb_filenames.erase(rgb_filenames.begin(),rgb_filenames.begin()+3);
        depth_filenames.erase(depth_filenames.begin(),depth_filenames.begin()+3);
        timestamp_filenames.erase(timestamp_filenames.begin(),timestamp_filenames.begin()+3);

        //Completamos la Direccion para que pueda ser leido directamente
        for(int i = 0; i < rgb_filenames.size();i++){
            rgb_filenames[i] = database_name + "/" + rgb_filenames[i];
            depth_filenames[i] = database_name + "/" + depth_filenames[i];
        }
    }
    else
        std::cout << "Couldnt Open a File\n";
}

std::string Dataset::getRGB_filename(int i)
{
    return rgb_filenames[i];
}

std::string Dataset::getDEPTH_filename(int i)
{
    return depth_filenames[i];
}

std::string Dataset::getTimestamp_filename(int i){
    return timestamp_filenames[i];
}

int Dataset::NoFrames()
{
    return rgb_filenames.size();
}

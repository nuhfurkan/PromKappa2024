#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <fstream>

#define CYCLE_SIZE 4096

struct ChunkContext {
    std::string promoter_name = "unnamed";
    std::string promoter = "";
    int len;
    int window_size = 30;
    int id_cycle = 0;
    bool export_kappaic = false;
    FILE* export_kappaic_file;
    bool export_kappaic_medium = false;
    FILE* export_kappaic_medium_file;
};

struct Flags {
    std::string output_file = "";
    std::string input_file = "";
    std::string dna = "";
    int window_size = 30;
    bool create_kappa_ic = false;
    bool record_medium = false;
    int len_dna = -1;
    std::string name = "";
};

__global__ void moveWindow(const char* input, float** output, int n, int window_size, int id_cycle) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float m = 0;
    float t = 0;

    if (tid <= n) {
        output[0][tid] = tid + id_cycle*(CYCLE_SIZE-29);
        output[1][tid] = tid + id_cycle*(CYCLE_SIZE-29);
    
        float counter = 0;

        for (int u = 1; u < window_size; u++) {
            for (int i = 0; i < window_size - u; i++) {
                if (input[tid + i] == input[tid + i + u]) {
                    m += 1;
                }
            }

            t += (m / double(window_size - u)) * 100.0;
            m = 0;
            
            if (input[tid + u - 1] == 'G' || input[tid + u - 1] == 'C') {
                counter += 1;
            }
        }

        output[0][tid] = counter/float(window_size);
        output[1][tid] = (100.0 - (t / float(window_size - 1)));
    }
}

void createPatternForChunk(ChunkContext context) {
    int id;
    cudaGetDevice(&id);

    //std::cout << context.promoter.length() << std::endl;

    char *a = NULL; // for inputs
    float **b = NULL; // for outputs
    
    cudaMallocManaged(&a, context.len * sizeof(char));
    cudaMallocManaged(&b, 2 * sizeof(float*));

    for (int i = 0; i < 2; ++i) {
        cudaMallocManaged(&b[i], context.len * sizeof(float));
    }

    cudaMemcpy(a, context.promoter.c_str(), context.len * sizeof(char), cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (context.len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    moveWindow<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, context.len, context.window_size, context.id_cycle);

    cudaDeviceSynchronize();

    if (context.export_kappaic) {
        for(int i = 0; i < context.len-context.window_size; i++) {
            fprintf(context.export_kappaic_file, "%f, %f\n", b[0][i], b[1][i]);
        }
    }
    if (context.export_kappaic_medium) {
        float gpc = 0;
        float ic = 0;
        for(int i = 0; i < context.len-context.window_size; i++) {
            gpc += b[0][i];
            ic += b[1][i];
        }
        
        fprintf(
            context.export_kappaic_medium_file, 
            "%s, %f, %f\n",  
            context.promoter_name.c_str(),
            gpc/(context.len - context.window_size), 
            ic/(context.len - context.window_size)
        );
    }

    cudaFree(a);
    for (int i = 0; i < 2; ++i) {
        cudaFree(b[i]);
    }
    cudaFree(b);
    return;
}

bool directoryExists(std::string directoryName) {
    struct stat info;
    if (stat(directoryName.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

FILE* createFile(std::string folder_name, std::string file_name) {
    if (!directoryExists(folder_name)) {
        if (mkdir(folder_name.c_str(), 0777) == -1) {
            perror("Error creating directory");
            NULL;
        }
    } 
    
    int length = std::snprintf(nullptr, 0, "%s/%s", folder_name.c_str(), file_name.c_str());
    char buffer[length + 1];
    std::sprintf(buffer, "%s/%s", folder_name.c_str(), file_name.c_str());

    // Create and open the text file in write mode
    FILE *file = fopen(buffer, "w");
    if (file == NULL) {
        perror("Error creating file");
        NULL;
    }
    return file;
}

std::vector<std::string> splitString(const char* input) {
    std::istringstream iss(input);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

void StartPattern(Flags thisFlag) {
    if (thisFlag.dna != "") {
        struct ChunkContext newChunkContext;
        newChunkContext.promoter = thisFlag.dna;
        newChunkContext.window_size = thisFlag.window_size;
        newChunkContext.export_kappaic = thisFlag.create_kappa_ic;
        newChunkContext.export_kappaic_medium = thisFlag.record_medium;
        newChunkContext.len = thisFlag.len_dna;
        if (thisFlag.name != "") {
            newChunkContext.promoter_name = thisFlag.name;
        }
        char* kappa_ic_file_name = (char*)malloc(strlen(newChunkContext.promoter_name.c_str()));
        strcpy(kappa_ic_file_name, newChunkContext.promoter_name.c_str());
        char kappa_ic_medium_file_name[] = "kappa_ic_medium";
        FILE* ic_file = createFile(thisFlag.output_file, kappa_ic_file_name);
        FILE* medium_file = createFile(thisFlag.output_file, kappa_ic_medium_file_name);

        if (ic_file && medium_file) {
            newChunkContext.export_kappaic_file = ic_file;
            newChunkContext.export_kappaic_medium_file = medium_file;

            createPatternForChunk(newChunkContext);

            fclose(ic_file);
            fclose(medium_file);
        } else {
            printf("Error: Failed to create output files.\n");
        }
    } else {
        std::string dna;
        std::vector<std::string> token;
        FILE* datafile = fopen(thisFlag.input_file.c_str(), "r");
        FILE* ic_file;
        FILE* medium_file;

        if (thisFlag.record_medium) {
            char kappa_ic_medium_file_name[] = "kappa_ic_medium";
            medium_file = createFile(thisFlag.output_file, kappa_ic_medium_file_name);
        }

        if (datafile != NULL) {
            char line[1000]; // Assuming a maximum line length of 1000 characters
            while (fgets(line, sizeof(line), datafile)) {
                if (line[0] == '>') {
                    if (token.size() != 0) {
                        struct ChunkContext newChunkContext;
                        newChunkContext.promoter = dna;
                        newChunkContext.window_size = thisFlag.window_size;
                        newChunkContext.export_kappaic = thisFlag.create_kappa_ic;
                        newChunkContext.export_kappaic_medium = thisFlag.record_medium;
                        newChunkContext.len = dna.length();
                        newChunkContext.promoter_name = token.at(1);
                        char* kappa_ic_file_name = (char*)malloc(strlen(newChunkContext.promoter_name.c_str()));
                        strcpy(kappa_ic_file_name, newChunkContext.promoter_name.c_str());
                        ic_file = createFile(thisFlag.output_file, kappa_ic_file_name);                    
                        
                        newChunkContext.export_kappaic_file = ic_file;
                        newChunkContext.export_kappaic_medium_file = medium_file;
                        
                        createPatternForChunk(newChunkContext);
                        
                        fclose(ic_file);
                    }
                    dna = "";
                    token = splitString(line);
                    //std::cout << token.at(1) << std::endl;
                } else {
                    dna += std::string(line);
                    if (dna.at(dna.size()-1) == '\n') {
                        dna = dna.substr(0, dna.size() - 1);
                    }
                }
            }
            struct ChunkContext newChunkContext;
            newChunkContext.promoter = dna;
            newChunkContext.window_size = thisFlag.window_size;
            newChunkContext.export_kappaic = thisFlag.create_kappa_ic;
            newChunkContext.export_kappaic_medium = thisFlag.record_medium;
            newChunkContext.len = dna.length();
            newChunkContext.promoter_name = token.at(1);
            char* kappa_ic_file_name = (char*)malloc(strlen(newChunkContext.promoter_name.c_str()));
            strcpy(kappa_ic_file_name, newChunkContext.promoter_name.c_str());
            ic_file = createFile(thisFlag.output_file, kappa_ic_file_name);
        
            newChunkContext.export_kappaic_file = ic_file;
            newChunkContext.export_kappaic_medium_file = medium_file;
            
            createPatternForChunk(newChunkContext);
        
            fclose(ic_file);            
            fclose(medium_file);
            fclose(datafile);
        } else {
            fprintf(stderr, "Unable to open file: %s\n", thisFlag.input_file.c_str());
        }   
    }
    return;
}

int main(int argc, char* argv[]) {
    struct Flags this_Flag;
    bool helpRequested = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            this_Flag.output_file = argv[i + 1];
        } else if (strcmp(argv[i], "-i") == 0  && i + 1 < argc) {
            this_Flag.input_file = argv[i + 1];
        } else if (strcmp(argv[i], "-dna") == 0 && i + 1 < argc) {
            this_Flag.dna = argv[i + 1];
            this_Flag.len_dna = this_Flag.dna.length();
            i++;
        } else if (strcmp(argv[i], "-ws") == 0 && i + 1 < argc) {
            this_Flag.window_size = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-name") == 0 && i + 1 < argc) {
            this_Flag.name = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-h") == 0) {
            helpRequested = true;
        } else if (strcmp(argv[i], "-m") == 0) {
            this_Flag.record_medium = true;
        } else if (strcmp(argv[i], "-ic") == 0) {
            this_Flag.create_kappa_ic = true;
        }

    }

    // Print help message if requested
    if (helpRequested) {
        printf("Options:\n");
        printf("  -i <file>            : Input file\n");
        printf("  -dna <dna string>    : Dna String as argument\n");
        printf("  -h                   : Display this help message\n");
        printf("  -o <file>            : Specify output folder\n");
        printf("  -m                   : Process for medium value\n");
        printf("  -ic                  : Save KappaIC\n");
        printf("  -name <str>          : name of promoter\n");
        printf("  -ws <int>            : window size\n");
        return 0;
    }

    StartPattern(this_Flag);

    return 0;
}
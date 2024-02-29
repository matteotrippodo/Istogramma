#include <iostream>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cstdlib>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
// Read the environment variables
int max_iter = std::stoi(std::getenv("TEXT_ITER")); // How many times iterate on the text folder
int test_iter =std::stoi(std::getenv("TEST_ITER")); // How many times repeat the execution of the function

void UpdateHistogramLetter(unordered_map<string, int>& hist,const string& text, const int ngram_length){
    // Check if the text has at least ngram_length letters in it
    if (text.size() > ngram_length) {

        // Iterate over the ngrams of letters in the text
        for (int i = 0; i < text.size() - ngram_length + 1; i++) {

            string letter_string;

            // Compose the next sequence of letters which made the ngram
            for (int j = i; j < i + ngram_length; j++) {
                letter_string += text[j];
            }

            // Check if the string contains only alphabetic chars
            if (std::all_of(letter_string.begin(), letter_string.end(),
                            [](char ch) { return std::isalpha(ch) != 0; })) {

            // Convert to lowercase the letters
                for (char &ch: letter_string) {
                    ch = std::tolower(ch);
                }
            // Update the histogram
                hist[letter_string]++;
            }
        }
    }
}

void UpdateHistogramWord(unordered_map<string, int>& hist,const string& text, const int ngram_length){

    // Create a stream of words
    istringstream stream(text);

    // Extract the words from the stream and put them in the vector
    vector<string> words(istream_iterator<std::string>(stream), {});

    // Check if the text has at least ngram_length words in it
    if (words.size() > ngram_length) {

        // Iterate over the ngrams of words
        for (int i = 0; i < words.size() - ngram_length + 1; ++i) {

            string word_string;

            // Compose the next ngram of words
            for (int j = i; j < i + ngram_length - 1; j++) {
                word_string += words[j] + " ";
            }
            word_string += words[i + ngram_length - 1];

            // Convert to lowercase the words
            for (char &ch: word_string) {
                ch = std::tolower(ch);
            }

            // Erase the invalid chars in the string
            word_string.erase(std::remove_if(word_string.begin(), word_string.end(),
                                             [](char ch) { return !std::isalpha(ch) && ch != ' '; }),
                              word_string.end());

            // Update the histogram
            hist[word_string]++;
        }
    }
}

void CreateHistogramV1(vector<unordered_map<string,int>>& hists, const int ngram_length){
    // Create the vector of texts
    vector<string> texts;

#pragma omp parallel shared(texts, max_iter, ngram_length, hists) default(none)
    {
        // Create the local versions of the histograms
        unordered_map<string, int> thread_letter_hist;
        unordered_map<string, int> thread_word_hist;

        // Iterate over the texts
        for (int k=0;k<max_iter;k++) {
            for (const auto &document: filesystem::directory_iterator("./Texts")) {

                // Obtain the path of the next file
                string document_path = document.path().string();
                ifstream file(document_path);
                string content;

                #pragma omp single
                {   // Open the file and read the document
                    if (file.is_open()) {

                        string line;
                        stringstream buffer;

                        buffer << file.rdbuf();
                        content = buffer.str();
                        texts.push_back(content);

                        // Close the file
                        file.close();
                    }
                    else {
                        printf("Impossible open the file: %s", document_path.c_str());
                    }
                }
            }
        }

        // Update the local version of the histograms
#pragma omp for nowait schedule(dynamic,1)
        for (int l = 0; l < texts.size(); l++) {
            UpdateHistogramWord(thread_word_hist, texts[l], ngram_length);
            UpdateHistogramLetter(thread_letter_hist, texts[l], ngram_length);
        }

        // Update the output histogram using the critical sections
#pragma omp critical (letter)
        {
            for (const auto &entry: thread_letter_hist) {
                hists[0][entry.first] += entry.second;
            }
        }

#pragma omp critical (word)
        {
            for (const auto &entry: thread_word_hist) {
                hists[1][entry.first] += entry.second;
            }
        }

   }// end parallel section
}

void CreateHistogramV2(vector<unordered_map<string,int>>& hists,const int ngram_length){

    // Count the number of texts in the folder
    int doc_count = 0;
    for (const auto &document: filesystem::directory_iterator("./Texts")) {
        doc_count++;
    }

# pragma omp parallel shared(doc_count, max_iter, ngram_length, hists) default(none)
    {
        // Create the local versions of the histograms
        unordered_map<string, int> thread_letter_histogram;
        unordered_map<string, int> thread_word_histogram;

        // Iterates over the folder of texts
        for (int k=0; k < max_iter; k++) {
# pragma omp for nowait schedule(dynamic,1)
            for (int i = 0; i < doc_count; i++) {

                // Obtain the path of the next file
                string document_path = "./Texts/" + std::to_string(i) + ".txt";
                ifstream file(document_path);

                // Open the file, extract the text and update the histograms
                if (file.is_open()) {

                    stringstream buffer;
                    buffer << file.rdbuf();
                    string content = buffer.str();

                    UpdateHistogramLetter(thread_letter_histogram, content, ngram_length);
                    UpdateHistogramWord(thread_word_histogram, content, ngram_length);

                    // Close the file
                    file.close();
                } else {
                    printf("Impossible open the file: %s", document_path.c_str());
                }
            }
        }

        // Update the output histograms using the critical sections
#pragma omp critical (letter)
        {
            for (const auto &entry: thread_letter_histogram) {
                hists[0][entry.first] += entry.second;
            }
        }

#pragma omp critical (word)
        {
            for (const auto &entry: thread_word_histogram) {
                hists[1][entry.first] += entry.second;
            }
        }

    }// end parallel section
}

void CreateHistogramV3(vector<unordered_map<string,int>>& hists,const int ngram_length){

    // Count the texts in the folder
    int doc_count = 0;
    for (const auto &document: filesystem::directory_iterator("./Texts")) {
        doc_count++;
    }

#pragma omp parallel shared(doc_count, max_iter, ngram_length, hists) default(none)
    {
        // Create the local version of the histograms and the local set of texts
        vector<string> texts;
        unordered_map<string, int> thread_word_histogram;
        unordered_map<string, int> thread_letter_histogram;

        // Iterate over the text folder
        for (int k=0;k< max_iter;k++) {
#pragma omp for nowait schedule(dynamic,1)
            for (int i = 0; i < doc_count; i++) {

                // Obtain the path of the next file
                std::string document_path ="./Texts/" + std::to_string(i) + ".txt";
                ifstream file(document_path);

                // Open the file, extract the text and close the file
                if (file.is_open()) {

                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string content = buffer.str();
                    texts.push_back(content);

                    file.close();
                } else {
                    printf("Impossible open the file: %s", document_path.c_str());
                }
            }
        }
        // Update the local histograms using the local set of texts
        for (int l = 0; l < texts.size(); l++) {
            UpdateHistogramWord(thread_word_histogram, texts[l], ngram_length);
            UpdateHistogramLetter(thread_letter_histogram, texts[l], ngram_length);
        }

        // Update the output histograms using the critical sections
#pragma omp critical (letter)
        {
            for (const auto &entry: thread_letter_histogram) {
                hists[0][entry.first] += entry.second;
            }
        }

#pragma omp critical (word)
        {
            for (const auto &entry: thread_word_histogram) {
                hists[1][entry.first] += entry.second;
            }
        }

    }// end parallel section
}

void CreateHistogramSequential(vector<unordered_map<string,int>>& hists, const int ngram_length){

    // Iterate over the folder of texts
    for (int k = 0; k < max_iter; k++) {
        for (const auto &document: filesystem::directory_iterator("./Texts")) {

            // Obtain the path of the next file
            string document_path = document.path().string();

            // Open the file, read the document and update the histograms
            ifstream file(document_path);
            if (file.is_open()) {

                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();

                UpdateHistogramLetter(hists[0], content, ngram_length);
                UpdateHistogramWord(hists[1], content, ngram_length);

                // Close the file
                file.close();

            } else {
                cerr << "Impossible open the file: " << document_path << endl;
            }
        }
    }

}

int main() {

#ifdef _OPENMP
    cout << "_OPENMP defined" << std::endl;
#endif

    // Read the enviroment variable regarding the n-gram size
    const int ngram_size = std::stoi(std::getenv("NGRAM_SIZE"));

    // Define the array of tests, one varying the number of thread the other varying the size of texts
    const int test_thread_array [16] = {1, 2, 4, 6, 8, 10, 12, 14,16,18, 20,22,24,26,28, 30};
    const int test_size_array [6] = {1,2, 4, 6, 8, 10};

    string histogram_item;
    switch(ngram_size){
        case 2:
            histogram_item = "bigram";
            break;
        case 3:
            histogram_item = "trigram";
            break;
        default:
            break;
    }
    // Define the log file and erase its previous content
    std::ofstream logfile("log.txt", std::ofstream::trunc);

    // Iterate over the array of tests
    //for(const int text_iter : test_size_array) {
    //    max_iter = text_iter;
    for(const int &thread_num : test_thread_array) {
        omp_set_num_threads(thread_num);

        cout << "----------- Num used threads " << omp_get_max_threads() << std::endl;
        logfile << "NUM THREADS " <<  omp_get_max_threads() << std::endl;
        //cout << "----------- Texts size " << start_size*max_iter << " MB " <<std::endl;
        //logfile << "TEXT SIZE " << start_size*max_iter << " MB " <<std::endl;

        //  SEQUENTIAL PART
        vector<double> seq_test_times;
        int total;

        cout << "SEQUENTIAL " << endl;
        // Iterate multiple times to evaluate the average performance
        for (int test_count = 0; test_count < test_iter; test_count++) {
            cout << "Test " << test_count << " --> ";

            // Call the sequential function and measure the execution time
            double start_time = omp_get_wtime();
            vector<unordered_map<string, int>> result_hist(2);
            CreateHistogramSequential(result_hist, ngram_size);
            double end_time = omp_get_wtime();

            cout << "  Execution time : " << end_time - start_time;
            seq_test_times.push_back(end_time - start_time);

            total = 0;
            // Count the total number of n-grams in the histograms
            for (const auto &element: result_hist[0]) {
                total += element.second;
            }
            cout << " ,   Total letters " << histogram_item << " count : " << total ;

            total = 0;
            for (const auto &element: result_hist[1]) {
                total += element.second;
            }
            cout << " ,   Total words " << histogram_item << " count : " << total << endl;

            // Count the number of different n-grams in the histograms
            cout << "Total of different letters " << histogram_item << " : " << result_hist[0].size() ;
            cout << " ,  Total of different words " << histogram_item << " : " <<  result_hist[1].size()  << endl;
        }

        // Evaluate the mean execution time
        double sum, seq_mean;
        sum = std::accumulate(seq_test_times.begin(), seq_test_times.end(), 0.0);
        seq_mean = sum / seq_test_times.size();
        std::cout << "Mean sequential execution time : " << seq_mean << std::endl;

        // PARALLEL PART
        vector<double> par1_test_times;

        cout << "PARALLEL v1" << endl;
        logfile << "PARALLEL v1: " ;

        // Iterate multiple times to evaluate the average performance
        for (int test_count = 0; test_count < test_iter; test_count++) {
            cout << "Test " << test_count << " --> ";

            // Call the first parallel function and measure the execution time
            double start_time = omp_get_wtime();
            vector<unordered_map<string, int>> result_hist(2);
            CreateHistogramV1(result_hist, ngram_size);
            double end_time = omp_get_wtime();

            cout << "  Execution time : " << end_time - start_time;
            par1_test_times.push_back(end_time - start_time);

            total = 0;
            // Count the total number of n-grams in the histograms
            for (const auto &element: result_hist[0]) {
                total += element.second;
            }
            cout << " ,   Total letters " << histogram_item << " count : " << total ;

            total = 0;
            for (const auto &element: result_hist[1]) {
                total += element.second;
            }
            cout << " ,   Total words " << histogram_item << " count : " << total << endl;

            // Count the number of different n-grams in the histograms
            cout << "Total of different letters " << histogram_item << " : " << result_hist[0].size();
            cout << " ,  Total of different words " << histogram_item << " : " <<  result_hist[1].size()  << endl;
        }

        // Evaluate the mean execution time
        double par1_sum, par1_mean;
        par1_sum = std::accumulate(par1_test_times.begin(), par1_test_times.end(), 0.0);
        par1_mean = par1_sum / par1_test_times.size();
        std::cout << "Mean parallelv1 execution time : " << par1_mean << std::endl;

        // Compute the speedup for the first parallel version
        cout << "Speedup : " << seq_mean / par1_mean << endl;
        logfile << "Speedup : " << seq_mean / par1_mean << endl;


        vector<double> par2_test_times;
        cout << "PARALLEL v2" << endl;
        logfile << "PARALLEL v2: " ;

        // Iterate multiple times to evaluate the average performance
        for (int test_count = 0; test_count < test_iter; test_count++) {
            cout << "Test " << test_count << " --> ";

            // Call the second parallel function and measure the execution time
            double start_time = omp_get_wtime();
            vector<unordered_map<string, int>> result_hist(2);
            CreateHistogramV2(result_hist, ngram_size);
            double end_time = omp_get_wtime();

            cout << "  Execution time : " << end_time - start_time;
            par2_test_times.push_back(end_time - start_time);

            total = 0;
            // Count the total number of n-grams in the histograms
            for (const auto &element: result_hist[0]) {
                total += element.second;
            }
            cout << " ,   Total letters " << histogram_item << " count : " << total ;

            total = 0;
            for (const auto &element: result_hist[1]) {
                total += element.second;
            }
            cout << " ,   Total words " << histogram_item << " count : " << total << endl;

            // Count the number of different n-grams in the histograms
            cout << "Total of different letters " << histogram_item << " : " << result_hist[0].size();
            cout << " ,  Total of different words " << histogram_item << " : " <<  result_hist[1].size()  << endl;
        }

        // Evaluate the mean execution time
        double par2_sum, par2_mean;
        par2_sum = std::accumulate(par2_test_times.begin(), par2_test_times.end(), 0.0);
        par2_mean = par2_sum / par2_test_times.size();
        std::cout << "Mean parallelv2 execution time : " << par2_mean << std::endl;

        // Compute the speedup for the second parallel version
        cout << "Speedup : " << seq_mean / par2_mean << endl;
        logfile << "Speedup : " << seq_mean / par2_mean << endl;

        vector<double> par3_test_times;

        cout << "PARALLEL v3 " << endl;
        logfile << "PARALLEL v3: " ;

        // Iterate multiple times to evaluate the average performance
        for (int test_count = 0; test_count < test_iter; test_count++) {
            cout << "Test " << test_count << " --> ";

            // Call the third parallel function and measure the execution time
            double start_time = omp_get_wtime();
            vector<unordered_map<string, int>> result_hist(2);
            CreateHistogramV3(result_hist, ngram_size);
            double end_time = omp_get_wtime();

            cout << "  Execution time : " << end_time - start_time;
            par3_test_times.push_back(end_time - start_time);

            total = 0;
            // Count the total number of n-grams in the histograms
            for (const auto &element: result_hist[0]) {
                total += element.second;
            }
            cout << " ,   Total letters " << histogram_item << " count : " << total ;

            total = 0;
            for (const auto &element: result_hist[1]) {
                total += element.second;
            }
            cout << " ,   Total words " << histogram_item << " count : " << total << endl;

            // Count the number of different n-grams in the histograms
            cout << "Total of different letters " << histogram_item << " : " << result_hist[0].size() ;
            cout << " ,  Total of different words " << histogram_item << " : " <<  result_hist[1].size()  << endl;
        }

        // Evaluate the mean execution time
        double par3_sum, par3_mean;
        par3_sum = std::accumulate(par3_test_times.begin(), par3_test_times.end(), 0.0);
        par3_mean = par3_sum / par3_test_times.size();
        std::cout << "Mean parallelv3 execution time : " << par3_mean << std::endl;

        // Compute the speedup for the third parallel version
        cout << "Speedup : " << seq_mean / par3_mean << endl;
        logfile << "Speedup : " << seq_mean / par3_mean << endl;
        logfile << "\n";

    }

    logfile.close();
    return 0;
}

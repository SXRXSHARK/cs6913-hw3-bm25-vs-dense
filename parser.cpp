#include <iostream>
#include <fstream>
#include <string>
#include <cctype>
#include <sstream>
#include <istream>
#include <map>
#include <vector>
#include <set>
#include <chrono>
#include <unordered_set>
#include <cstdio> // For std::remove

std::string line;
std::string word;
int count = 0;
int temp_number = 0; // temp file number
long long total_words = 0; // for stats: all words added together
std::map<std::string, std::vector<std::pair<int, int>>> inverted_index;

int main() {
    std::ifstream file("collection.tsv");
    if (!file) {
        std::cout << "Could not open the file\n";
        return 1;
    }

    std::ifstream subset ("msmarco_passages_subset.tsv");
    std::unordered_set<std::string> keep; // keep is an empty {}
    std::string subset_line; // subset_line is a string that will hold each line of the subset file
    while (std::getline(subset, subset_line)) { // grab the line from the subset file
        size_t tab = subset_line.find('\t'); // find the tab which is between the ID and the text
        if (tab != std::string::npos) { // this is to skip lines with no tab
            keep.insert(subset_line.substr(0, tab)); // take the ID before the tab and insert it into the keep set
        }
    }
    std::cout << "Subset ids loaded: " << keep.size() << "\n";
    auto start = std::chrono::steady_clock::now(); // start the timer
    auto total_start = std::chrono::steady_clock::now(); // total timer, never restarts
    std::ofstream pageFile("pagetable.csv"); // docID and length
    if (!pageFile) {
        std::cout << "Could not create page table file\n";
        return 1;
    }
    pageFile << "DocID" << "," << "word_count" << "\n"; 
    
    while (std::getline(file, line)) {
        // grab the line from the file
        size_t tab = line.find('\t'); // find the tab which is between the ID and the text
        if (tab == std::string::npos) continue; // this is to skip lines with no tab
        std::string DocID = line.substr(0, tab); // take the ID before the tab
        if (keep.find(DocID) == keep.end()) continue; // not in the 1M subset, skip it
        std::string text = line.substr(tab + 1); // take the text after the tab
        for (char& c : text) { // it goes through letter by letter
            if (isalnum(static_cast<unsigned char>(c))) c = static_cast<char>(tolower(c)); // keep letters and numbers, make lowercase, isalnum checks if the character is alphanumeric, tolower converts it to lowercase
            else c = ' '; // everything else becomes a space like punctuations
        }
        std::stringstream ss(text); // it puts the clean text into string stream
        int word_count = 0;
        std::map<std::string, int> passage_terms;
        while (ss >> word) {
            // to keep giving the next word
                word_count++; // counts the number of words in the passage
                passage_terms[word]++; // counts the frequency of each word in the passage
        }
        for (auto& p : passage_terms) inverted_index[p.first].emplace_back(count, p.second); // p.first is the word, p.second is the frequency, count is the docID. It adds the docID and frequency to the inverted index for each word in the passage.
        pageFile << DocID << "," << word_count << "\n";
        total_words += word_count; // adds the number of words in this passage to the total word count
        count++; // increments the docID for the next passage
        if (count % 750000 == 0) { // 750000 passages per temp file, so that the temp files are not too big. This is a good balance between memory usage and speed.
            std::ofstream tempFile("temp_" + std::to_string(temp_number) + ".csv");
            tempFile << "Term,DocID,Frequency\n";
            for (auto& entry : inverted_index) {
                for (auto& posting : entry.second) {
                    tempFile << entry.first << "," << posting.first << "," << posting.second << "\n";
                }
            }
            inverted_index.clear(); // after each temp file is written, so temp files no longer repeat earlier postings. Essentially, it clears the memory.
            std::cout << "temp_" << temp_number << " seconds: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "\n";
            start = std::chrono::steady_clock::now(); // restart the timer for the next temp file
            temp_number++; // next file number
        }
    }
    // write whatever is left as the last temp file. This is the reminder of the passages that were not written to a temp file yet.
    std::ofstream tempFile("temp_" + std::to_string(temp_number) + ".csv");
    tempFile << "Term,DocID,Frequency\n";
    for (auto& entry : inverted_index) {
        for (auto& posting : entry.second) {
            tempFile << entry.first << "," << posting.first << "," << posting.second << "\n";
        }
    }
    std::cout << "temp_" << temp_number << " seconds: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "\n";
    start = std::chrono::steady_clock::now();
    std::ofstream statsFile("stats.csv");
    statsFile << "Number of DocIDs," << count << "\n";
    statsFile << "Average Passage Length," << (double) total_words / count << "\n"; // avg passage length = total_words ÷ number of passages
    std::cout << "Total lines read: " << count << "\n";
    pageFile.close();
    tempFile.close();
    statsFile.close();
    std::cout << "Total seconds: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count() << "\n";
    return 0;
}
//my own notes
// I used comma so that it is easier to read the csv files in excel. 
// I used a map to store the inverted index because it allows for efficient lookups and insertions.
// I used a vector of pairs to store the document ID and frequency for each term, which allows for easy iteration and storage of multiple postings for each term.
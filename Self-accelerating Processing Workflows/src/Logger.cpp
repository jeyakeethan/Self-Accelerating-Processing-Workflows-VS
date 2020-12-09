#include<sstream>
#include "Logger.h"

Logger Logger::instance;

Logger::Logger() {}

void Logger::open(const string& logFile) {
    instance.fileStream.open(logFile.c_str(), ios::app);
}
void Logger::close() {
    instance.fileStream.close();
}
void Logger::write(const string& message) {
    ostream& stream = instance.fileStream;
    stream << message;
}

void Logger::clearLogs(const string& logFile) {
    if (Logger::isOpen()) {
        Logger::close();
    }
    instance.fileStream.open(logFile.c_str());
    ostream& stream = instance.fileStream;
    stream << "";
}
bool Logger::isOpen() {
    return instance.fileStream.is_open();
}

void Logger::writeToFile(string path, const string& txt) {
    ofstream stream;
    stream.open(path, ios::out);
    stream << txt;
    stream.close();
}

string Logger::readFromFile(string path) {
    ifstream stream;
    stream.open(path);
    string s;
    getline(stream,s);
    stream.close();
    return s;
}
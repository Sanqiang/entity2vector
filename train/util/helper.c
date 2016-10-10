#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



struct eqstr
{
    bool operator()(const char* s1, const char* s2) const
    {
        return (s1 == s2) || (s1 && s2 && strcmp(s1, s2) == 0);
    }
};

char* concat(char *s1, char *s2)
{
    char *result = (char *)malloc((strlen(s1)+strlen(s2)+1)* sizeof(char));//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000
float* expTable;
void init_util(){
    //populate exp precomputing table
    int i;
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
}
#include "parameters.h"

int FLOW_BACK;   // reserve_check
int BORDER_SIZE; // 边界大小
int MIN_DIST;    // 特征点的最小距离
int MAX_NUM;     // 维持特征点数目
int STEREO;      // 使用双目

void setParameters()
{
    FLOW_BACK = 1;   // 开启reserve_check
    BORDER_SIZE = 1; // 边界大小设为1
    MIN_DIST = 10;
    MAX_NUM = 100;
    STEREO = 1;
} 
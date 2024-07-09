__kernel void convolve(const __global uint *const input,
                       __constant uint* const mask,
                       __global uint *const output,
                       const int inputWidth,
                       const int maskWidth){
    const int x=get_global_id(0);
    const int y=get_global_id(1);
 
 
    uint sum=0;
    if(x%inputWidth < inputWidth - maskWidth + 1 && x/inputWidth < inputWidth - maskWidth + 1){
        for(int r=0;r < maskWidth; r++){
            const int idxIntpm=(y+r) * inputWidth+x;    //x作为全局坐标，这是每一行的第一个
    
            for(int c=0;c<maskWidth;c++){             //卷积核与之相乘
                sum+=mask[(r*maskWidth) + c]*input[idxIntpm + c];   //边缘会出现错误
            }
        }
        output[x%inputWidth + (x/inputWidth)*(inputWidth - maskWidth + 1)]=sum;
    }
}

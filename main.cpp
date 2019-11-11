#include "tensor/XTensor.h"          // 引用XTensor定义的头文件
#include "tensor/XGlobal.h"
#include "tensor/XCall.h"
#include "tensor/core/CHeader.h"
#include "tensor/core/arithmetic/Sum.h"
//#include "tensor/core/arithmetic/Sum.h"
#include "tensor/function/FHeader.h"
#include "network/XNet.h"
#include <iostream>
//#include "NiuTensor-master/source/sample/regression/FNNReg.cpp"
#include <vector>
//#include "NiuTensor-master/source/sample/fnnlm/FNNLM.cpp"
using namespace nts;
struct Model{
    XTensor w1;
    XTensor b1;
    XTensor w2;
    XTensor w3;
    XTensor b2;
    XTensor b3;
    XTensor emb;
    int batchSize = 49;
    int devID = -1;
    void Init(){
        InitTensor2D(&emb, 7, 3, X_FLOAT, devID);
        InitTensor2D(&w1,2,1,X_FLOAT,devID);
        InitTensor2D(&w2,2,1,X_FLOAT,devID);
        InitTensor2D(&w3,3,1,X_FLOAT,devID);
        InitTensor1D(&b1,3,X_FLOAT,devID);
        InitTensor1D(&b2,3,X_FLOAT,devID);
        InitTensor1D(&b3,1,X_FLOAT,devID);

        emb.SetVarFlag();
        w1.SetVarFlag();
        w2.SetVarFlag();
        w3.SetVarFlag();
        b1.SetVarFlag();
        b2.SetVarFlag();
        b3.SetVarFlag();

        emb.SetDataRand(-5,5);
        w1.SetDataRand(-5,5);
        w2.SetDataRand(-5,5);
        b1.SetDataRand(-5,5);
        b2.SetDataRand(-5,5);
        b3.SetDataRand(-5,5);


    }
    void Update(Model grad,float learningRate){
        emb = Sum(emb, grad.emb, -learningRate);
        w1 = Sum(w1, grad.w1, -learningRate);
        w2 = Sum(w2, grad.w2, -learningRate);
        w3 = Sum(w2, grad.w2, -learningRate);
        b1 = Sum(b1, grad.b1, -learningRate);
        b2 = Sum(b2, grad.b2, -learningRate);
        b3 = Sum(b2, grad.b2, -learningRate);
    }
    void Train(XTensor &X, XTensor &Y){
        XNet autoDiffer;
        Model grad;
        float learningRate = 0.001;
        InitTensor(&grad.w1,&w1);
        InitTensor(&grad.w2,&w2);
        InitTensor(&grad.emb,&emb);
        InitTensor(&grad.b1,&b1);
        InitTensor(&grad.b2,&b2);
        grad.devID = devID;
        for(int epoch = 0; epoch < 30; epoch++){
            if(w1.grad != NULL)
                w1.grad->SetZeroAll();
            if(w2.grad != NULL)
                w2.grad->SetZeroAll();
            if(emb.grad != NULL)
                emb.grad->SetZeroAll();
            if(b1.grad != NULL)
                b1.grad->SetZeroAll();
            if(b2.grad != NULL)
                b2.grad->SetZeroAll();
            XTensor output = Forward(X);

            printf("%d %d %d %d %d %d \n%d\n",output.GetDim(0),output.GetDim(1),output.order,Y.GetDim(0),Y.GetDim(1),Y.order,_IsSameShaped(&output,&Y));
//            output.Dump(stderr);
            XTensor lossTensor = CrossEntropy(output,Y);
            autoDiffer.Backward(lossTensor);
            Update(grad,learningRate);
        }
    }
    XTensor Forward(XTensor &X){
        XTensor embedding = Gather(emb,X);
        embedding = Transpose(embedding,1,2);
        XTensor h1,h2;
        XTensor b = Unsqueeze(b1,0,batchSize);
        XTensor t = MatrixMul(embedding,w1);
        h1 = HardTanH(MatrixMul(embedding,w1)+b);
        b = Unsqueeze(b2,0,batchSize);
        h2 = HardTanH(MatrixMul(embedding,w2)+b);
        t = Sigmoid(h1+h2);
        t = Squeeze(t);
        return MatrixMul(t,w3)+Unsqueeze(b3,0,batchSize);
    }
};
int main(int argc, const char ** argv)
{
    TensorList trainDataX;
    TensorList trainDataY;
    XTensor *X = NewTensor2D(7*7,2,X_INT);
    XTensor *Y = NewTensor2D(7*7,1,X_INT);
    int row = 0;
    int data[49][2];
    int tag[49][1];
    for(int i = 0; i <= 6; i++){
        for(int j = 0; j <= 6; j++){
            data[row][0] = i;
            data[row][1] = j;
            tag[row][0] = i^j;
//            X->Set2D(i,row,0);
//            X->Set2D(j,row,1);
//            Y->Set1D(i^j,row);
            row++;
        }
    }
    X->SetData(data,49*2);
    Y->SetData(tag,49);
//    *Y = Unsqueeze(*Y,1,1);
//    for(int i = 0; i <= 48; i++) {
//        for (int j = 0; j <= 1; j++) {
//            printf("%f ",((float*)X->data)[i+j]);
//        }
//        printf("\n");
//    }
//    printf("%d",X->dimSize[1]);
    Model model;
    model.Init();
    model.Train(*X,*Y);


    return 0;
}
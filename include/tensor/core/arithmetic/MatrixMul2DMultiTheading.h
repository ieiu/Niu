/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northestern University.
* All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#ifndef __MATRIXMUL2DMULTITHEADING_H__
#define __MATRIXMUL2DMULTITHEADING_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication for a block (x1,y1) - (x2,y2)
where (x1,y1) is the upper-left corner and (x2,y2) is the bottom-right corner
*/
void _MatrixMul2DMultiTheading(TensorList * args);

} // namespace nts(NiuTrans.Tensor)

#endif // __MATRIXMUL2DMULTITHEADING_H__
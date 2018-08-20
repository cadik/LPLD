#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

#define PATCHCONSTANT 16
#define PATCHCONSTANTSQUARE 256
#define DICTIONARY_SIZE_CONSTANT 32



real_t getVariance(real_t subImage[PATCHCONSTANTSQUARE], real_t meanValue){
	int numOfItems = PATCHCONSTANTSQUARE;
	real_t copyImage[PATCHCONSTANTSQUARE];
	for(int i = 0; i < PATCHSIZE; i++){
		for(int j = 0; j < PATCHSIZE; j++){
			copyImage[i + PATCHSIZE*j] =subImage[i + PATCHSIZE*j];
		}
	}
	real_t temp =0;
	for(int i = 0; i < numOfItems; i++) {
        temp += (meanValue-(copyImage[i]))*(meanValue-(copyImage[i]));
    }
	return temp/numOfItems;
}


real_t zr_function(int num){

	if( num ==0){
		return 0.7071067;
	}
	return 1.0;
}




__kernel void bow(
	__global real_t * output,
	const __global real_t * m_diff,
	__constant real_t * dictionary,
	__constant int * zigzag
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	//const int rowOffset = get_global_id(1) * IMAGE_W;
    //const int my = get_global_id(0) + rowOffset;

	int patchCord_X = x * STRIDE + 1;
    int patchCord_Y = y * STRIDE + 1;

	real_t patchVector[PATCHCONSTANT];
	//DCT_Descriptor(patchCord_X, patchCord_Y, m_diff, zigzag, patchVector );
	int patchSize = PATCHSIZE;
	int halfPathcSize =patchSize / 2;
	real_t subImage[PATCHCONSTANTSQUARE];
	int patchSquareCount = patchSize * patchSize;

	int tempx = 0;
	int tempy =0;
	if((patchCord_X + halfPathcSize - (IMAGE_W*2 - 1)) > 0){
		tempx = patchCord_X + halfPathcSize - (IMAGE_W*2 - 1);
	}
	if((patchCord_Y + halfPathcSize - (IMAGE_H*2 - 1)) > 0){
		tempy = patchCord_Y + halfPathcSize - (IMAGE_H*2 - 1);
	}
	

	int pixCordTmp_X = patchCord_X - tempx;
    int pixCordTmp_Y = patchCord_Y -tempy;
	tempx = 0;
	tempy = 0;
	if((-1*(pixCordTmp_X - halfPathcSize)) > 0){
		tempx = -1*(pixCordTmp_X - halfPathcSize);
	}
	if((-1*(pixCordTmp_Y - halfPathcSize)) > 0){
		tempy = -1*(pixCordTmp_Y - halfPathcSize);
	}
	

    pixCordTmp_X = pixCordTmp_X +tempx;
    pixCordTmp_Y = pixCordTmp_Y +tempy;

    int top_x = pixCordTmp_X - (halfPathcSize - 1);
    int top_y = pixCordTmp_Y - (halfPathcSize - 1);

	for(int i = 0; i < patchSize; i++){
		for(int j = 0; j < patchSize; j++){
			subImage[i + patchSize*j] = m_diff[top_x+i + IMAGE_W*2*(top_y+j)];
		}
	}
	

	unsigned int descriptorDim = PATCHSIZE;
	real_t meanValue = 0;
	if (LUM_INVARIANT_PATCH) {
		for(int i = 0; i < patchSize; i++){
			for(int j = 0; j < patchSize; j++){
				meanValue = meanValue + subImage[i + patchSize*j];
			}
		}
		meanValue = meanValue/ (patchSize * patchSize);
		for(int i = 0; i < patchSize; i++){
			for(int j = 0; j < patchSize; j++){
				subImage[i + patchSize*j] = subImage[i + patchSize*j] - meanValue;
			}
		}
	}
	if (CONSTRAST_INVARIANT_PATCH) {
		real_t subImageVar = 0;
		if (LUM_INVARIANT_PATCH){
			subImageVar = sqrt(max(0.005, getVariance(subImage,meanValue)));
		}else{
			for(int i = 0; i < patchSize; i++){
				for(int j = 0; j < patchSize; j++){
					meanValue = meanValue + subImage[i + patchSize*j];
				}
			}
			subImageVar = sqrt(max(0.005, getVariance(subImage,meanValue)));
		}
		real_t subImageContrastVar = (1.0 /subImageVar);
		for(int i = 0; i < patchSize; i++){
			for(int j = 0; j < patchSize; j++){
				subImage[i + patchSize*j] = subImage[i + patchSize*j] * subImageContrastVar;
			}
		}
	}
	
	real_t cnst = 3.1415926;
	cnst = cnst / (2*patchSize);
	real_t divConstant = sqrt(2.0/patchSize) * sqrt(2.0/patchSize);
	

	real_t dctImage[PATCHCONSTANTSQUARE];
	for(int u = 0; u <patchSize; u++){
		for(int v = 0; v < patchSize; v++){
				
			real_t temp =0;
			for(int i = 0; i < patchSize; i++){
				for(int j = 0; j < patchSize; j++){
					temp = temp + zr_function(i) * zr_function(j) * cos(cnst * u * (2*i +1)) * cos(cnst * v * (2*j +1)) *  subImage[i + patchSize*j] ;
				}
			}
			dctImage[u+ v*patchSize] = divConstant * temp;
			
		}
	}
	
	int dctOffset = 0;

	if(LUM_INVARIANT_PATCH) {
        dctOffset = 1;
        if (descriptorDim <= 4)
            descriptorDim = descriptorDim - 1;
    }
	 for(unsigned int dctCoeff = 0; dctCoeff < descriptorDim; dctCoeff++) {
		int x = zigzag[dctCoeff*2 + dctOffset];
        int y = zigzag[dctCoeff*2 +1 + dctOffset];

		patchVector[dctCoeff] = dctImage[x + y*patchSize];
	 }
	 
	output[x+y*IMAGE_W] = patchVector[4];
	real_t patchDescriptor[PATCHCONSTANT * DICTIONARY_SIZE_CONSTANT];
	for(int i = 0; i < PATCHSIZE; i++){
		for(int j = 0; j< DICTIONARY_SIZE; j++){
			patchDescriptor[j + i*DICTIONARY_SIZE] = patchVector[i];
		}
	}
	
	real_t tmp[PATCHCONSTANT * DICTIONARY_SIZE_CONSTANT];
	for(int i = 0; i < DICTIONARY_SIZE ; i++){
		for(int j = 0; j< PATCHSIZE ; j++){
			real_t t1 = patchDescriptor[i + j*DICTIONARY_SIZE];
			real_t t2 = dictionary[i + j*DICTIONARY_SIZE];

			tmp[i + j*DICTIONARY_SIZE] = sqrt((t1-t2)*(t1-t2));
		}
	}
	//output[x+y*IMAGE_W] = tmp[4 + 32*8];
	real_t distance[DICTIONARY_SIZE_CONSTANT];
	for(int i = 0; i < DICTIONARY_SIZE ; i++){
		distance[i] = 0;
	}
	for(int i = 0; i < DICTIONARY_SIZE ; i++){
		for(int j = 0; j< PATCHSIZE  ; j++){
			distance[i] += tmp[i + j*DICTIONARY_SIZE];
		}
	}
	//output[x+y*IMAGE_W] = distance[4];
	real_t min = 500;
	int loc = 0;
	for(int i = 0; i < DICTIONARY_SIZE ; i++){
		if(distance[i]<min){
			min= distance[i];
			loc = i;
		}
	}

	output[x+y*IMAGE_W] = loc;
}


#define NEIGHBOR_CONSTANT 16

__kernel void bow2(
	const __global real_t * expanded_Img,
	const __global real_t * H,
	__constant real_t * gaussian,
	__global real_t * output

){
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int neighborInterval = 2*NEIGHBOUR_RADIUS+1;

	real_t tmpROI[(2*NEIGHBOR_CONSTANT+1)*(2*NEIGHBOR_CONSTANT+1)];
	for(int i = 0; i< neighborInterval;i++){
		for(int j = 0; j< neighborInterval; j++){
			tmpROI[i + j*neighborInterval] = expanded_Img[(x+i) + (y+j)*(NCOLS +neighborInterval-1)];
		}
	}
	real_t tmpHeader[32];
	for(int i =0; i< 32; i++){
		tmpHeader[i] = 0;
	}


	 for(int roiX = 0; roiX < neighborInterval ; roiX++) {
		for(int roiY = 0; roiY < neighborInterval; roiY++) {
			int Hselect = (int) tmpROI[roiX + roiY * neighborInterval];

			tmpHeader[Hselect] += gaussian[roiX + roiY * neighborInterval];
		}
	}
	for(int i =0; i< 32; i++){
		output[(x+ y*NCOLS) + (NCOLS*NROWS)*i] = tmpHeader[i];
	}
	//output[(x+ y*NCOLS)] = tmpHeader[4];
}





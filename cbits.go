package tflite

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"fmt"
	"unsafe"
	//"image"
	//"path/filepath"
	"bufio"
	"os"
	"sort"

	//"github.com/anthonynsimon/bild/imgio"
	//"github.com/anthonynsimon/bild/transform"
	"github.com/Unknwon/com"
	//"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
)

// accelerator modes
const (
	CPUMode        = 0
	GPUMode        = 1
	DSPMode        = 2
	VisualCoreMode = 3
)

// struct for keeping hold of predictor
type PredictorData struct {
	ctx   C.PredictorContext
	mode  int
	batch int
}

// make mode and batch public
func (pd *PredictorData) Inc() {
	pd.mode++
	pd.batch++
}

func NewPredictorData() *PredictorData {
	return &PredictorData{}
}

func New(model string, mode, batch int) (*PredictorData, error) {

	// fetch model file
	modelFile := model
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	// set device for acceleration
	switch mode {
	case 0:
		SetUseCPU()
	case 1:
		SetUseGPU()
	case 2:
		SetUseDSP()
	case 3:
		SetUseVisualCore()
	default:
		SetUseCPU()
	}

	return &PredictorData{
		ctx: C.NewTflite(
			C.CString(modelFile),
			C.int(batch),
			C.int(mode),
		),
		mode:  mode,
		batch: batch,
	}, nil
}

func SetUseCPU() {
	C.SetModeTflite(C.int(CPUMode))
}

func SetUseGPU() {
	C.SetModeTflite(C.int(GPUMode))
}

func SetUseDSP() {
	C.SetModeTflite(C.int(DSPMode))
}

func SetUseVisualCore() {
	C.SetModeTflite(C.int(VisualCoreMode))
}

func init() {
	C.InitTflite()
}

func Predict(p *PredictorData, data []byte) error {

	// check for null imagedata
	if len(data) == 0 {
		return fmt.Errorf("image data is empty")
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	C.PredictTflite(p.ctx, ptr)

	return nil
}

// return predicted class (top-1)
func ReadPredictionOutput(p *PredictorData, labelFile string) (string, error) {

	batchSize := p.batch
	if batchSize == 0 {
		return "", errors.New("null batch")
	}

	predLen := int(C.GetPredLenTflite(p.ctx))
	if predLen == 0 {
		return "", errors.New("null predLen")
	}
	length := batchSize * predLen
	if p.ctx == nil {
		return "", errors.New("empty predictor context")
	}
	cPredictions := C.GetPredictionsTflite(p.ctx)

	if cPredictions == nil {
		return "", errors.New("empty predictions")
	}

	slice := (*[1 << 15]float32)(unsafe.Pointer(cPredictions))[:length:length]

	var labels []string
	f, err := os.Open(labelFile)
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(slice) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(slice[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	top1 := features[0][0]
	return top1.GetClassification().GetLabel(), nil

}

func Close(p *PredictorData) {
	C.DeleteTflite(p.ctx)
}

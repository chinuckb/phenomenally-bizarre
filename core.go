// main.go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
)

// ModelParams holds the HMM parameters
type ModelParams struct {
	NComponents int         `json:"n_components"`
	StartProb   []float64   `json:"startprob"`
	TransMat    [][]float64 `json:"transmat"`
	Means       [][]float64 `json:"means"`
	Covars      [][][]float64 `json:"covars"`
}

// LoadParams loads model parameters from JSON file
func LoadParams(filename string) (*ModelParams, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var params ModelParams
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&params)
	return &params, err
}

// GaussianPDF calculates the probability density for Gaussian distribution
func GaussianPDF(x, mean, variance float64) float64 {
	exponent := math.Exp(-math.Pow(x-mean, 2) / (2 * variance))
	return (1 / math.Sqrt(2 * math.Pi * variance)) * exponent
}

// PredictNextState predicts the next state based on current observation
func PredictNextState(params *ModelParams, currentObservation float64, currentState int) int {
	stateProbs := make([]float64, params.NComponents)
	for nextState := 0; nextState < params.NComponents; nextState++ {
		mean := params.Means[nextState][0]
		variance := params.Covars[nextState][0][0]
		obsProb := GaussianPDF(currentObservation, mean, variance)
		stateProbs[nextState] = params.TransMat[currentState][nextState] * obsProb
	}

	// Choose state with maximum probability
	nextState := 0
	maxProb := stateProbs[0]
	for i := 1; i < params.NComponents; i++ {
		if stateProbs[i] > maxProb {
			maxProb = stateProbs[i]
			nextState = i
		}
	}
	return nextState
}

func main() {
	// Load model parameters from JSON file
	params, err := LoadParams("hmm_model_params.json")
	if err != nil {
		log.Fatalf("Error loading HMM parameters: %v", err)
	}

	// Example observation sequence (replace with actual data)
	observations := []float64{1.0, 0.8, 1.2, 0.4, 0.6, 0.3}
	currentState := 0

	fmt.Println("Observations and Predicted States:")
	for _, obs := range observations {
		nextState := PredictNextState(params, obs, currentState)
		fmt.Printf("Observation: %.2f, Predicted State: %d\n", obs, nextState)
		currentState = nextState
	}
}

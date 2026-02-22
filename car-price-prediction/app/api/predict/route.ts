import { NextResponse } from 'next/server';
import {
  preparePredictionInput,
  applyScaling,
  predict,
  type PredictionInput,
} from '@/lib/ml-model';
import { getOrTrainModel, addPredictionToHistory } from '@/lib/data-store';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const input = body as PredictionInput;
    
    // Validate input
    if (!input.year || !input.presentPrice || input.drivenKms === undefined) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: year, presentPrice, drivenKms' },
        { status: 400 }
      );
    }
    
    // Validate ranges
    if (input.year < 1990 || input.year > new Date().getFullYear() + 1) {
      return NextResponse.json(
        { success: false, error: 'Invalid year. Must be between 1990 and current year + 1' },
        { status: 400 }
      );
    }
    
    if (input.presentPrice < 0) {
      return NextResponse.json(
        { success: false, error: 'Present price must be positive' },
        { status: 400 }
      );
    }
    
    if (input.drivenKms < 0) {
      return NextResponse.json(
        { success: false, error: 'Kilometers driven must be positive' },
        { status: 400 }
      );
    }
    
    // Get or train model
    const model = getOrTrainModel();
    
    // Prepare and scale input features
    const rawFeatures = preparePredictionInput(input);
    const scaledFeatures = applyScaling(rawFeatures, model.scalingParams);
    const predictedPrice = predict(scaledFeatures, model.weights);
    
    // Ensure non-negative prediction
    const finalPrice = Math.max(0, predictedPrice);
    
    // Add to prediction history
    const historyRecord = addPredictionToHistory({
      input: {
        year: input.year,
        presentPrice: input.presentPrice,
        drivenKms: input.drivenKms,
        fuelType: input.fuelType,
        sellerType: input.sellerType,
        transmission: input.transmission,
        owner: input.owner,
      },
      predictedPrice: finalPrice,
    });
    
    return NextResponse.json({
      success: true,
      prediction: {
        price: finalPrice,
        priceFormatted: `₹${finalPrice.toFixed(2)} Lakh`,
        predictionId: historyRecord.id,
      },
      modelInfo: {
        version: model.version,
        trainedAt: model.trainedAt,
      },
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to make prediction' },
      { status: 500 }
    );
  }
}

import { NextResponse } from 'next/server';
import {
  preprocessData,
  standardizeFeatures,
  predictBatch,
} from '@/lib/ml-model';
import { getOrTrainModel, getAllCarData } from '@/lib/data-store';

// GET /api/charts - Get visualization data
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const chartType = searchParams.get('type') || 'all';
    
    const model = getOrTrainModel();
    const data = getAllCarData();
    const { features, targets } = preprocessData(data);
    const { scaledFeatures } = standardizeFeatures(features);
    const predictions = predictBatch(scaledFeatures, model.weights);
    
    const chartData: Record<string, unknown> = {};
    
    // Actual vs Predicted
    if (chartType === 'all' || chartType === 'actualVsPredicted') {
      chartData.actualVsPredicted = targets.map((actual, i) => ({
        actual,
        predicted: Math.max(0, predictions[i]),
      })).slice(0, 100);
    }
    
    // Price by Fuel Type
    if (chartType === 'all' || chartType === 'fuelType') {
      const fuelTypeData = data.reduce((acc, car) => {
        if (!acc[car.Fuel_Type]) {
          acc[car.Fuel_Type] = { total: 0, count: 0 };
        }
        acc[car.Fuel_Type].total += car.Selling_Price;
        acc[car.Fuel_Type].count += 1;
        return acc;
      }, {} as Record<string, { total: number; count: number }>);
      
      chartData.avgPriceByFuelType = Object.entries(fuelTypeData).map(([fuel, d]) => ({
        fuel,
        avgPrice: Number((d.total / d.count).toFixed(2)),
      }));
    }
    
    // Price by Year
    if (chartType === 'all' || chartType === 'year') {
      const yearData = data.reduce((acc, car) => {
        if (!acc[car.Year]) {
          acc[car.Year] = { total: 0, count: 0 };
        }
        acc[car.Year].total += car.Selling_Price;
        acc[car.Year].count += 1;
        return acc;
      }, {} as Record<number, { total: number; count: number }>);
      
      chartData.avgPriceByYear = Object.entries(yearData)
        .map(([year, d]) => ({
          year: parseInt(year),
          avgPrice: Number((d.total / d.count).toFixed(2)),
        }))
        .sort((a, b) => a.year - b.year);
    }
    
    // Residuals
    if (chartType === 'all' || chartType === 'residuals') {
      chartData.residuals = targets.map((actual, i) => ({
        predicted: Math.max(0, predictions[i]),
        residual: Number((actual - predictions[i]).toFixed(2)),
      })).slice(0, 100);
    }
    
    // Price by Transmission
    if (chartType === 'all' || chartType === 'transmission') {
      const transmissionData = data.reduce((acc, car) => {
        if (!acc[car.Transmission]) {
          acc[car.Transmission] = { total: 0, count: 0 };
        }
        acc[car.Transmission].total += car.Selling_Price;
        acc[car.Transmission].count += 1;
        return acc;
      }, {} as Record<string, { total: number; count: number }>);
      
      chartData.avgPriceByTransmission = Object.entries(transmissionData).map(([trans, d]) => ({
        transmission: trans,
        avgPrice: Number((d.total / d.count).toFixed(2)),
      }));
    }
    
    // Price by Seller Type
    if (chartType === 'all' || chartType === 'sellerType') {
      const sellerData = data.reduce((acc, car) => {
        if (!acc[car.Selling_type]) {
          acc[car.Selling_type] = { total: 0, count: 0 };
        }
        acc[car.Selling_type].total += car.Selling_Price;
        acc[car.Selling_type].count += 1;
        return acc;
      }, {} as Record<string, { total: number; count: number }>);
      
      chartData.avgPriceBySellerType = Object.entries(sellerData).map(([seller, d]) => ({
        sellerType: seller,
        avgPrice: Number((d.total / d.count).toFixed(2)),
      }));
    }
    
    // Feature Importance
    if (chartType === 'all' || chartType === 'featureImportance') {
      chartData.featureImportance = model.featureImportance;
    }
    
    // Price Distribution (histogram data)
    if (chartType === 'all' || chartType === 'priceDistribution') {
      const prices = data.map(car => car.Selling_Price);
      const min = Math.floor(Math.min(...prices));
      const max = Math.ceil(Math.max(...prices));
      const bucketSize = Math.max(1, Math.ceil((max - min) / 10));
      
      const buckets: Record<string, number> = {};
      prices.forEach(price => {
        const bucketIndex = Math.floor((price - min) / bucketSize);
        const bucketLabel = `${min + bucketIndex * bucketSize}-${min + (bucketIndex + 1) * bucketSize}`;
        buckets[bucketLabel] = (buckets[bucketLabel] || 0) + 1;
      });
      
      chartData.priceDistribution = Object.entries(buckets).map(([range, count]) => ({
        range,
        count,
      }));
    }
    
    // Kilometers vs Price scatter
    if (chartType === 'all' || chartType === 'kmVsPrice') {
      chartData.kmVsPrice = data.slice(0, 100).map(car => ({
        kms: car.Driven_kms,
        price: car.Selling_Price,
        name: car.Car_Name,
      }));
    }
    
    return NextResponse.json({
      success: true,
      chartData,
      modelVersion: model.version,
    });
  } catch (error) {
    console.error('Chart data error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to generate chart data' },
      { status: 500 }
    );
  }
}

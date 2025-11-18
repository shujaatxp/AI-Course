using Microsoft.ML;

public static class TrainModel
{
    public static void Build()
    {
        var mlContext = new MLContext();
        var data = mlContext.Data.LoadFromTextFile<HouseData>(
            "C:\\Personal\\AICourse\\MLNet\\MLNet\\house_data.csv", hasHeader: true, separatorChar: ',');

        var pipeline = mlContext.Transforms
            .Concatenate("Features", "Size", "Bedrooms")
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Price",maximumNumberOfIterations:5));

        var model = pipeline.Fit(data);
        mlContext.Model.Save(model, data.Schema, "model.zip");

        Console.WriteLine("✅ ML.NET model trained & saved!");
    }
}

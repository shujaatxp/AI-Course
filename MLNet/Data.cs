using Microsoft.ML.Data;

public class HouseData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }
}

public class HousePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

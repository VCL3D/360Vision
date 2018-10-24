# Single Image Depth Metrics
Python (3.5) script for computing Single Image Depth Error Metrics presented in:  
Eigen, D., Puhrsch, C. and Fergus, R., 2014. 
"_Depth map prediction from a single image using a multi-scale deep network_". 
In Advances in neural information processing systems (pp. 2366-2374).

The script was used for calculating the Depth Estimation errors presented in:

N. Zioulis, A. Karakottas, D. Zarpalas, P. Daras, 
"_OmniDepth: Dense Depth Estimation for Indoors Spherical Panoramas_", 
European Conference on Computer Vision (ECCV), Munich, Germany, 8 – 14 September 2018

Unfortunately, we identified a bug in the script before uploading it. 
The bug is fixed; however the tables in the above publication are slightly modified when running this script. 
Specifically, the following tables of the paper are now:

## Table 2
     
|Network       | Abs. Rel. | Sq. Rel | RMSE  | RMSE(log) | δ < 1.25  | δ < 1.25^2  | δ < 1.25^3  |
|--------------|-----------|---------|-------|-----------|-----------|-------------|-------------|
|UResNet       |   0.806   | 0.0324  |0.2804 | 0.1158    | 0.9361    |     0.99    |   0.997     |      
|RectNet       |   0.0687  | 0.02448 |0.2432 | 0.0999    | 0.9583    |     0.9936  |   0.998     |
|              |           |         |**Equirectangular**       |           |           |             |             |
|Godard et al. |   0.6221  | 1.3497  |1.6196 | 1.0092    | 0.2308    |     0.4353  |   0.5977    |
|Laina et al.  |   0.2972  | 0.2713  |0.7863 | 0.3649    | 0.4955    |     0.7842  |   0.9209    | 
|Liu et al.    |   0.3272  | 0.3224  |0.9826 | 0.4223    | 0.3914    |     0.7095  |   0.8836    |
|              |           |         | **Cubemaps**       |           |           |             |             |
|Godard et al. |   0.5455  | 0.9126  | 1.437 | 0.7855    | 0.2969    |     0.5265  |   0.6894    |
|Laina et al.  |   0.3536  | 0.5261  | 1.0513| 0.3822    | 0.4671    |     0.7602  |   0.9113    | 
|Liu et al.    |   0.2792  | 0.2785  | 0.8259| 0.3448    | 0.5373    |     0.8084  |   0.8269    |   

      
## Table 3

|Network       | Abs. Rel. | Sq. Rel | RMSE  | RMSE(log) | δ < 1.25  | δ < 1.25^2  | δ < 1.25^3  |
|--------------|-----------|---------|-------|-----------|-----------|-------------|-------------|
|Godard et al. | 0.07457   | 0.2655  | 0.6487| 0.2535    | 0.3407    |   0.5452    |   0.6781    |
|Laina et al.  | 0.02457   | 0.09415 | 0.2913| 0.096     | 0.6719    |   0.88785   |   0.9542    |
|Liu et al.    | 0.02886   | 0.1989  | 0.3389| 0.1106    | 0.6737    |   0.8355    |   0.9282    |

As you can see all the results are a little better, but their relative performance has not changed.

## Running the script
The script assumes that the Ground Truth files (in _.exr_ format) are in the same directory as the Prediction files (also in _.exr_ format),
and have the same names with the Prediction files suffixed with "_pred".
  
Just run:  
python SingleImageDepthMetrics.py --path <_dir/to/depth/files_>

## Dependencies
* OpenCV (python)
* Numpy

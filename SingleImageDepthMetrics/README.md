# Single Image Depth Metrics
Python (3.5) script for computing Single Image Depth Error Metrics presented in:  
Eigen, D., Puhrsch, C. and Fergus, R., 2014. 
"_Depth map prediction from a single image using a multi-scale deep network_". 
In Advances in neural information processing systems (pp. 2366-2374).

# Running
The script assumes that the Ground Truth files (in _.exr_ format) are in the same directory as the Prediction files (also in _.exr_ format),
and have the same names with the Prediction files suffixed with "_pred".
  
Just run:  
python SingleImageDepthMetrics.py --path <_dir/to/depth/files_>

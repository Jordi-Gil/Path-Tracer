echo "GPU_NO_BVH"
cd GPU_NO_BVH
sh job_1GPU.sh

echo "CPU_OMP_BVH"
cd ../CPU_OMP_BVH
sh job.sh

echo "CPU_OMP_BVH_IT"
cd ../CPU_OMP_BVH_IT
sh job.sh

echo "CPU_OMP_NO_BVH"
cd ../CPU_OMP_NO_BVH
sh job.sh

echo "CPU_SEC_BVH"
cd ../CPU_SEC_BVH
sh job.sh

echo "CPU_SEC_BVH_IT"
cd ../CPU_SEC_BVH_IT
sh job.sh

echo "CPU_SEC_NO_BVH"
cd ../CPU_SEC_NO_BVH
sh job.sh

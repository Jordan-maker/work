projectName="B"
date="28Mar22" 
destinyFolder="../Moriond2021"

mkdir complete

listNames=(proc11p1 proc11p2 bucket9 bucket10 bucket11)

for i in ${listNames[@]}; do 
    
    cd ${destinyFolder}
    
    projectFolder="${projectName}_${i}_${date}"
    tar -xvf ${projectFolder}.tar 
    cd $projectFolder
    
    for k in 0 1; do
    	if [ -d "sub0$k" ]; then
    	    cd sub0$k
    	    scp * ${destinyFolder}/complete
    	fi
    	cd ..
    done
    
    cd ${destinyFolder}
    rm -fr $projectFolder
        
done



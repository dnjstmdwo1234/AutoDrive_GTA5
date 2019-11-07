<h1>AutoDrive_GTA5</h1>

<p>
<h2>preface</h2> 
<h5>This project is a private project in the Futuristic_car and robot capstone design, a Department of Software Convergence course at Kyunghee University. To help with the project, I referenced the YouTube guide and some code from sentex. Annotated the referenced part of the code.</h5></p>
<br>
</p><h2>Project Goals</h2>
<h5>When I select any point on the map, the Car follow traffic signals and arrive at destination without accident.</h5></p>
<br>
</p>
<h3>1. Detection</h3>
<h4>(1) Lane detection using OpenCv(fail) -> Road detection using OpenCv</h4> 
<h5> <1> ROI Settings </h5>
<h5> <2> Change RGB to grayscale </h5>
<h5> <3> Set the RGB range of the road and treat it all white if it is not within the range </h5>
<h4>(2) Human and other Object recognition Using Tensorflow Object Detection API</h4>
<br>
<h3>2. Control</h3>
<h4>(1) Control Algorithm after Road Detection </h4>
  <h5><1> Divide the top 50 pixels of height from the transformed image into three equal pieces and extract each image.</h5>
    <h5><2> Obtain and store the average rgb of each image.</h5>
      <h5><3> Find the image that is closest to the road color of the rgb average extracted from the third-class image.</h5>
        <h5><4> If the left is closest to the road color, turn left; if the middle is closest to the road color, go straight; and if the right is closest to the road color, turn right. </h5>
<br>

<h3>3. Navigation & Path search</h3>
</p>

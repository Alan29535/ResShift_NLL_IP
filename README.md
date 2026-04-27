<h2 align="center">Observations and Analysis</h2>

<p>
We observe that <b>texture regions exhibit higher uncertainty</b> than smooth regions, making them more difficult for the model to learn. 
The corresponding implementation is available in <code>reprod_results2 copy.ipynb</code>.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/73b1024b-a43e-4e89-9c19-b632c2095426" width="500"/>
</p>

<p>
To further support this observation, we measure the <b>KL divergence</b> between each reverse step and its corresponding forward process using 16 ImageNet images.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/63f1b5ba-ca2c-4c8b-8a66-47d3f9290d27" width="420"/>
</p>

<p>
We further hypothesize that the <b>reconstruction error</b> arises from the mismatch between the variance learned in the reverse process (optimized via VLB) and the predefined variance in the forward process. 
To verify this, we track the variance evolution in both processes using <b>ResShift</b>. 
The implementation is available in <code>variance_compare_three_models.ipynb</code>.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f8540d3a-96eb-4426-9350-d49fef0f8413" width="420"/>
</p>

<p>
Moreover, due to limited computational resources, we are unable to provide results of our proposed method at this stage. However, we release the full experimental code for exploring the limitations of the ResShift model in Github.
</p>

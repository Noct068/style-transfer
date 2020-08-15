tf.ENV.set('WEBGL_PACK', false);
var h = 378;
class Main {
  constructor() {
      //load model
      
      this.styleButton = document.getElementById('style-button');
      this.styleButton.disabled = true;
        this.loadMobilenet().then(model => {
          this.styleNet = model;
        }).finally(() => this.enableStylize());
    // load transformer
    
    this.styleButton.disabled = true;
        this.loadTransformer().then(model => {
          this.transformNet = model;
        }).finally(() => this.enableStylize());

    this.startStylingProcess();

    Promise.all([
      this.loadMobilenet(),
      this.loadTransformer(),
    ]).then(([styleNet, transformNet]) => {
      console.log('Loading Complete');
      this.styleNet = styleNet;
      this.transformNet = transformNet;
      this.enableStylize()
    });
  }
    // load model function
  async loadMobilenet() {
    if (!this.mobileStyleNet) {
      this.mobileStyleNet = await tf.loadGraphModel(
        'models/inception/model.json');
    }
    return this.mobileStyleNet;
  }
   // load transformer function
  async loadTransformer() {
    if (!this.separableTransformNet) {
      this.separableTransformNet = await tf.loadGraphModel(
        'models/transformer/model.json'
      );
    }

    return this.separableTransformNet;
  }

  
  

  startStylingProcess() {
    // Initialize images
    this.contentImg = document.getElementById('content-img');
    this.contentImg.onerror = () => {
      alert("Error loading " + this.contentImg.src + ".");
    }
    this.styleImg = document.getElementById('style-img');
    this.styleImg.onerror = () => {
      alert("Error loading " + this.styleImg.src + ".");
    }
    this.stylized = document.getElementById('stylized');
    this.styleImg.height = 256;
    this.styleImg.style.width = 256 + "px";
    
    // style strength ratio     
    
    this.styleRatio = 1.0
    this.ratioSlider = document.getElementById('stylized-img-ratio');
    this.ratioSlider.oninput = (evt) => {
      this.styleRatio = evt.target.value/100.;
    }
    this.styleButton = document.getElementById('style-button');
    this.styleButton.onclick = () => {
      h = this.contentImg.width;
        
    this.contentImg.height = 256;
    this.contentImg.style.width = 256 + "px";
      this.styleButton.disabled = true;
      this.startTransfer().finally(() => {
        this.enableStylize();
          this.contentImg.style.width = h + "px";
        this.stylized.style.height = 256 + "px";
        this.stylized.style.width = h + "px";
      });
    }; 
    

    // initialize inputs
    this.contentSelect = document.getElementById('content-select');
    this.styleSelect = document.getElementById('style-select');
    this.styleSelect.onchange = (evt) => this.uploadAndLoadImage(this.styleImg, evt.target.value);
    this.styleSelect.onclick = () => this.styleSelect.value = '';
  }
  // upload or load content and style images

  uploadAndLoadImage(element, selectedValue) {
    if (selectedValue === 'file') {
      console.log('file selected');
      this.fileSelect.onchange = (evt) => {
        const f = evt.target.files[0];
        const fileReader = new FileReader();
        fileReader.onload = ((e) => {
          element.src = e.target.result;
            
        });
        fileReader.readAsDataURL(f);
        this.fileSelect.value = '';
      }
      this.fileSelect.click();
        
    } else {
      element.src = 'images/' + selectedValue + '.jpg';
        
    }
  }

  enableStylize() {
    this.styleButton.disabled = false;
    
    this.styleButton.textContent = 'Stylize';
  }

  async startTransfer() {
    await tf.nextFrame();
    this.styleButton.textContent = 'In process...';
    await tf.nextFrame();
    let pipeline = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
    })
    if (this.styleRatio !== 1.0) {
      this.styleButton.textContent = 'Just a moment...';
      await tf.nextFrame();
      const identityPipeline = await tf.tidy(() => {
        return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
      })
      const stylePipeline = pipeline;
      pipeline = await tf.tidy(() => {
        const stylePipelineScaled = stylePipeline.mul(tf.scalar(this.styleRatio));
        const identityPipelineScaled = identityPipeline.mul(tf.scalar(1.0-this.styleRatio));
        return stylePipelineScaled.addStrict(identityPipelineScaled)
      })
      stylePipeline.dispose();
      identityPipeline.dispose();
    }
    this.styleButton.textContent = 'Stylizing image...';
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
      return this.transformNet.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), pipeline]).squeeze();
    })
    await tf.browser.toPixels(stylized, this.stylized);
    pipeline.dispose();
    stylized.dispose();
  }
}
window.addEventListener('load', () => new Main());
var fileContent = document.getElementById('file-content');
var fileStyle = document.getElementById('file-style');
fileContent.onchange = (event) => {
            var image = document.getElementById('content-img');
            image.style.width = ''; 
            image.src = URL.createObjectURL(event.target.files[0]);

}
fileStyle.onchange = (event) => {
                var image = document.getElementById('style-img');
                image.src = URL.createObjectURL(event.target.files[0]);
}

    
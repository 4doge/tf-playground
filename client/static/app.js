const app = new Vue({
  el: "#app",
  data: {
    imgToPredict: null,
    imgToPredictName: null,
    imgToPredictPreview: null,
    loading: true,
    model: null,
    classes: null,
    predictions: []
  },
  async mounted() {
    this.model = await tf.loadGraphModel("model/model.json");
    this.classes = {
      0: "Normal",
      1: "Tuberculosis"
    };
    this.loading = false;
  },
  methods: {
    async predict() {
      if (this.imgToPredict) {
        this.loading = true;
        const tensor = tf.browser
          .fromPixels(this.$refs.preview, 3)
          .resizeNearestNeighbor([224, 224])
          .expandDims()
          .toFloat()
          .reverse(-1);
        let predictions = await this.model.predict(tensor).data();
        this.loading = false;
        this.predictions = Array.from(predictions)
          .map((p, i) => {
            return {
              probability: p,
              className: this.classes[i]
            };
          })
          .sort(function(a, b) {
            return b.probability - a.probability;
          });
      }
    },
    handleImage(e) {
      if (e.target.files && e.target.files[0]) {
        this.imgToPredict = e.target.files[0];
        this.imgToPredictName = this.imgToPredict.name;
        const reader = new FileReader();
        reader.onload = e => {
          this.imgToPredictPreview = e.target.result;
        };
        reader.readAsDataURL(this.imgToPredict);
      }
    }
  }
});

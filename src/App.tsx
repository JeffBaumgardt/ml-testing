import * as React from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSSD from "@tensorflow-models/coco-ssd";

import classData from "./classData";
import { images } from "./images";

const getImageDimensions = (
  img: HTMLImageElement,
  maxHeight: number,
  maxWidth: number
): Promise<{ height: number; width: number }> => {
  return new Promise((resolve, reject) => {
    img.onload = () => {
      const ratio = img.width / img.height;

      let newWidth = maxWidth;
      let newHeight = Math.ceil(newWidth / ratio);
      if (newHeight > maxHeight) {
        newHeight = maxHeight;
        newWidth = newHeight * ratio;
      }
      resolve({
        height: newHeight,
        width: newWidth,
      });
    };
    img.onerror = (err) => {
      reject(err);
    };
  });
};

type ImageLoadTime = Record<string, number>;

export default function App() {
  const [imageCount, setImageCount] = React.useState(0);
  const [imageLoadTimes, setImageLoadTimes] = React.useState<ImageLoadTime>({});
  const [image, setImage] = React.useState<HTMLImageElement | null>(null);
  const [canvasDimension, setCanvasDimension] = React.useState({
    height: 0,
    width: 0,
  });
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [modalLoadStart, setModalLoadStart] = React.useState(0);
  const [modalLoadEnd, setModalLoadEnd] = React.useState(0);
  const [firstTimeStamp, setFirstTimeStamp] = React.useState(0);
  const [secondTimeStamp, setSecondTimeStamp] = React.useState(0);

  const [model, setModel] = React.useState<cocoSSD.ObjectDetection>();

  const loadModel = React.useCallback(async () => {
    const model = await cocoSSD.load();
    setModel(model);
    setModalLoadEnd(window.performance.now());
  }, []);

  const buildRectangle = (predictions: cocoSSD.DetectedObject[]) => {
    const ctx = canvasRef.current!.getContext("2d");

    if (!ctx) return;

    ctx.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);

    ctx.lineWidth = 2;
    ctx.textBaseline = "bottom";
    ctx.font = "14px sans-serif";

    predictions.forEach((prediction) => {
      console.log(prediction);
      const guessText = `${prediction.class}`;

      ctx.strokeStyle = classData[guessText as keyof typeof classData];
      const textWidth = ctx.measureText(guessText).width;
      const textHeight = Number(ctx.font);

      ctx.strokeRect(
        prediction.bbox[0],
        prediction.bbox[1],
        prediction.bbox[2],
        prediction.bbox[3]
      );
      ctx.fillStyle = "white";
      ctx.fillRect(
        prediction.bbox[0] - ctx.lineWidth / 2,
        prediction.bbox[1],
        textWidth + ctx.lineWidth,
        -textHeight
      );
      ctx.fillStyle = "#fc0303";
      ctx.fillText(
        `${guessText[0].toUpperCase() + guessText.slice(1)} - Score: ${
          prediction.score
        }`,
        prediction.bbox[0],
        prediction.bbox[1]
      );
    });
    setSecondTimeStamp(window.performance.now());
  };

  const runPredictions = React.useCallback(async () => {
    if (!model || !image) {
      return;
    }
    setFirstTimeStamp(window.performance.now());
    const predictions = await model.detect(image);
    buildRectangle(predictions);
  }, [image, model]);

  const loadImage = React.useCallback(async () => {
    const image = new Image();
    const randomImage = images[Math.floor(Math.random() * images.length)];

    image.src = randomImage;

    getImageDimensions(image, 600, 600).then(({ height, width }) => {
      setCanvasDimension({
        height,
        width,
      });
      setImage(image);
    });
    setImageCount((imageCount) => imageCount + 1);
  }, []);

  React.useEffect(() => {
    setModalLoadStart(window.performance.now());
    tf.setBackend("webgl");
    console.log(tf.getBackend());
    tf.ready().then(() => {
      loadModel();
    });
  }, [loadModel]);

  React.useEffect(() => {
    if (image) {
      setFirstTimeStamp(0);
      setSecondTimeStamp(0);
      runPredictions();
    }
  }, [image, runPredictions]);

  React.useEffect(() => {
    if (image && firstTimeStamp > 0 && secondTimeStamp > 0) {
      setImageLoadTimes((prevLoadTimes) => ({
        ...prevLoadTimes,
        [image.src]: secondTimeStamp - firstTimeStamp,
      }));
    }
  }, [firstTimeStamp, image, secondTimeStamp]);

  const averageTime = React.useMemo(() => {
    const times = Object.keys(imageLoadTimes).reduce(
      (prevImageTime, currentImage) => {
        const currentTime = imageLoadTimes[currentImage];
        return prevImageTime + currentTime;
      },
      0
    );
    console.log(times, imageLoadTimes);
    return times / imageCount;
  }, [imageLoadTimes, imageCount]);

  return (
    <div style={{ margin: 10 }}>
      {model ? (
        <div style={{ marginBottom: 10 }}>
          <button onClick={loadImage}>Load Random Image</button>
          <span style={{ marginLeft: 10 }}>
            Modal Load Time: {(modalLoadEnd - modalLoadStart).toFixed(2)}ms
          </span>
        </div>
      ) : (
        <div>Loading model...</div>
      )}
      {image ? (
        <>
          <div className="image-container">
            {/* eslint-disable-next-line jsx-a11y/img-redundant-alt */}
            <img src={image.src} alt="test-image" className="image-position" />
            <canvas
              ref={canvasRef}
              width={canvasDimension.width}
              height={canvasDimension.height}
              className="image-position"
            />
          </div>
          <p>
            Image inference time:{" "}
            {(secondTimeStamp - firstTimeStamp).toFixed(2)}ms
          </p>
          <p>Average inference time: {averageTime.toFixed(2)}ms</p>
        </>
      ) : null}
    </div>
  );
}

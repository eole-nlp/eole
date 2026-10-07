import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: '⚙️ Open Models',
    description: (
      <>
        Train from scratch or bring supported Hugging Face checkpoints for language generation, translation, vision, OCR, and speech.
      </>
    ),
  },
  {
    title: '🧱 Simplicity and Modularity',
    description: (
      <>
        Validated YAML configurations, reusable model components, LoRA fine-tuning, and native translation scorers support experimentation.
      </>
    ),
  },
  {
    title: '💨 Speed and Efficiency',
    description: (
      <>
        Serve through chat APIs and use optional CUDA kernels, quantization, compilation, and native MTP on compatible models and hardware.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

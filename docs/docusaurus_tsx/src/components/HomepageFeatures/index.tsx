import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: '⚙️ Open Models',
    Svg: null,
    description: (
      <>
        This project, although not as broad in scope as some, aims at maximizing code reusability while supporting diverse architectures and pretrained models.
        The challenge here is to factorize while not over-complexifyng things too much.
      </>
    ),
  },
  {
    title: '🧱 Simplicity and Modularity',
    Svg: null,
    description: (
      <>
        A single entry-point to call runnables and tools.
        Pre-defined converters, extendable modules and architectures.
      </>
    ),
  },
  {
    title: '💨 Speed and Efficiency',
    Svg: null,
    description: (
      <>
        Developed on reasonable hardware.
        Aimed at making models accessible in restricted resources or frugal environments. 
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      {/* <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div> */}
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

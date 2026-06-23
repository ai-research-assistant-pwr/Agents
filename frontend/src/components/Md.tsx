import ReactMarkdown from 'react-markdown';

interface Props {
  children: string;
  className?: string;
}

export default function Md({ children, className }: Props) {
  return (
    <div className={`md-prose${className ? ` ${className}` : ''}`}>
      <ReactMarkdown>{children}</ReactMarkdown>
    </div>
  );
}

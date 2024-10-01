import winston from 'winston';
import colors from 'colors/safe';

const customFormat = winston.format.printf(({ level, message, timestamp }) => {
    const coloredTimestamp = colors.grey(timestamp);
    let coloredLevel: string;

    switch (level) {
        case 'error':
            coloredLevel = colors.red(level);
            break;
        case 'warn':
            coloredLevel = colors.yellow(level);
            break;
        case 'info':
            coloredLevel = colors.green(level);
            break;
        case 'verbose':
            coloredLevel = colors.cyan(level);
            break;
        case 'debug':
            coloredLevel = colors.blue(level);
            break;
        default:
            coloredLevel = colors.white(level);
    }

    return `${coloredTimestamp} ${coloredLevel}: ${message}`;
});

export const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({
            format: 'YYYY-MM-DD HH:mm:ss'
        }),
        customFormat
    ),
    transports: [
        new winston.transports.Console()
    ]
});
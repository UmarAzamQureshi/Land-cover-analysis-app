# LandCover Change Detection - Modern UI

## Overview

The LandCover Change Detection platform now features a modern, responsive web interface with dark/light theme support and comprehensive functionality for land cover analysis.

## Features

### 🎨 Modern Design
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Dark/Light Theme Toggle**: Switch between themes with persistent preference storage
- **Smooth Animations**: Enhanced user experience with CSS animations and transitions
- **Accessibility**: WCAG compliant with keyboard navigation and screen reader support

### 📊 Dashboard
- **Real-time Status**: Live system status monitoring
- **Status Cards**: Visual indicators for data availability, pipeline status, and outputs
- **Quick Actions**: One-click access to common operations
- **Activity Log**: Recent system activities and notifications

### 📁 File Management
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **File Validation**: Automatic validation of .tif/.tiff files
- **Progress Tracking**: Real-time upload progress with visual indicators
- **File Management**: View, remove, and manage uploaded files

### 🔬 Analysis Pipeline
- **Pipeline Control**: Start, monitor, and manage analysis processes
- **Status Monitoring**: Real-time pipeline status updates
- **Configuration Display**: View current pipeline settings
- **Progress Tracking**: Monitor analysis progress

### 📤 Output Management
- **Output Browser**: Browse and download generated files
- **File Information**: Display file sizes and metadata
- **Direct Downloads**: One-click file downloads
- **Output Organization**: Categorized output display

## Interface Sections

### 1. Dashboard Tab
- **System Status Overview**: Current pipeline and data status
- **Quick Actions**: Start analysis, refresh status
- **Status Indicators**: Visual cards showing system health
- **Recent Activity**: Latest system events

### 2. Upload Files Tab
- **Drag & Drop Zone**: Intuitive file upload area
- **File List**: Manage uploaded files
- **Upload Progress**: Visual progress indicators
- **File Validation**: Automatic format checking

### 3. Analysis Tab
- **Pipeline Status**: Current analysis status
- **Data Availability**: Check for required data files
- **Analysis Controls**: Start and monitor analysis
- **Configuration**: View pipeline settings

### 4. Outputs Tab
- **Output Files**: Browse generated files
- **Download Links**: Direct file downloads
- **File Metadata**: Size and format information
- **Refresh Controls**: Update output list

## Theme System

### Light Theme
- Clean, professional appearance
- High contrast for readability
- Suitable for well-lit environments
- Default theme for new users

### Dark Theme
- Reduced eye strain in low-light conditions
- Modern, sleek appearance
- Energy efficient for OLED displays
- Automatically saves user preference

## Responsive Design

### Desktop (1200px+)
- Full feature set available
- Multi-column layouts
- Hover effects and animations
- Detailed status information

### Tablet (768px - 1199px)
- Optimized layouts for touch
- Simplified navigation
- Touch-friendly buttons
- Responsive grids

### Mobile (320px - 767px)
- Single-column layouts
- Touch-optimized interface
- Simplified navigation tabs
- Essential features only

## Accessibility Features

### Keyboard Navigation
- Tab navigation through all interactive elements
- Enter/Space key activation
- Escape key for closing modals
- Arrow key navigation in lists

### Screen Reader Support
- Semantic HTML structure
- ARIA labels and descriptions
- Alt text for images
- Status announcements

### Visual Accessibility
- High contrast mode support
- Reduced motion preferences
- Focus indicators
- Color-blind friendly design

## Browser Support

### Supported Browsers
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Required Features
- CSS Grid and Flexbox
- CSS Custom Properties (variables)
- ES6+ JavaScript
- Fetch API

## Performance Optimizations

### Loading Performance
- Minimal CSS and JavaScript
- Optimized images and icons
- Lazy loading for non-critical content
- Efficient DOM manipulation

### Runtime Performance
- Debounced API calls
- Efficient event handling
- Minimal reflows and repaints
- Optimized animations

## Usage Instructions

### Getting Started
1. **Access the Interface**: Navigate to the application URL
2. **Upload Data**: Use the Upload Files tab to add .tif/.tiff files
3. **Start Analysis**: Use the Analysis tab to begin processing
4. **Monitor Progress**: Check the Dashboard for real-time status
5. **Download Results**: Use the Outputs tab to access results

### File Requirements
- **Format**: .tif or .tiff files only
- **Size**: No strict limits, but consider upload time for large files
- **Content**: Geospatial raster data for land cover analysis

### Best Practices
- **Data Preparation**: Ensure files are properly georeferenced
- **File Naming**: Use descriptive names for easier management
- **Batch Processing**: Upload multiple files for comprehensive analysis
- **Regular Monitoring**: Check status regularly during long processes

## Troubleshooting

### Common Issues

#### Upload Problems
- **File Format**: Ensure files are .tif or .tiff format
- **File Size**: Large files may take time to upload
- **Network Issues**: Check internet connection for upload failures

#### Analysis Issues
- **Data Requirements**: Ensure sufficient data files are uploaded
- **System Resources**: Analysis may be resource-intensive
- **Pipeline Status**: Check dashboard for current status

#### Display Issues
- **Theme Problems**: Try refreshing the page or clearing cache
- **Responsive Issues**: Check browser window size and zoom level
- **JavaScript Errors**: Ensure JavaScript is enabled

### Support
- Check the main README.md for API documentation
- Review system logs for detailed error information
- Ensure all dependencies are properly installed

## Development

### File Structure
```
static/
├── index.html          # Main interface
├── styles.css          # Additional styles
├── upload.html         # Legacy upload interface
└── dashboard.html      # Legacy dashboard interface
```

### Customization
- **Themes**: Modify CSS variables in styles.css
- **Layout**: Adjust grid and flexbox properties
- **Animations**: Customize CSS animations and transitions
- **Functionality**: Extend JavaScript for additional features

### Contributing
- Follow existing code style and patterns
- Test on multiple devices and browsers
- Ensure accessibility compliance
- Update documentation for new features
